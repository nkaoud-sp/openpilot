#!/usr/bin/env python3
"""
meeting_actions.py - turn a recorded meeting (mp3, m4a, wav, mp4, ...) into
speaker-attributed, timestamped minutes and action points. Built for long
(1-2 hour), bilingual meetings, e.g. Arabic + English. Everything runs
locally with open-source models; nothing is sent to a cloud service.

Pipeline
  1. Who-spoke-when (diarization) ....... sherpa-onnx: pyannote-3.0 segmentation +
                                          3D-Speaker ERes2Net embeddings
  2. Language per speech piece .......... small Whisper model, limited to --languages
  3. Speech-to-text, language fixed ..... faster-whisper (large-v3-turbo), batched
     per piece, with word timestamps
  4. Words + speakers -> utterances ..... this script
  5. Topics / decisions / action items .. local LLM: llama.cpp (Qwen3-8B GGUF) or Ollama.
     Long meetings are analysed in parts, then duplicates are merged.
  6. The LLM cites utterance numbers; this script maps them to real timestamps,
     so timestamps are never invented by the model.

Models are downloaded on first use, after asking you (use --yes to skip the question).
Slow steps are checkpointed next to the audio file, so an interrupted run resumes.

Examples
  python meeting_actions.py meeting.m4a --languages ar,en
  python meeting_actions.py meeting.mp3 --languages ar,en --speakers 5 --names "A=Nizar,B=Sara"
  python meeting_actions.py --transcript meeting.mp3.transcript.json --minutes-language Arabic
  python meeting_actions.py meeting.mp3 --llm ollama --ollama-model qwen3:8b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults (all open-source, ungated)

DEFAULT_WHISPER = "large-v3-turbo"           # good Arabic + English; "large-v3" is a bit better, 2x slower
DEFAULT_LID_MODEL = "base"                   # small multilingual Whisper, used only to detect the language
DEFAULT_GGUF_REPO = "Qwen/Qwen3-8B-GGUF"     # Apache-2.0, multilingual incl. Arabic
DEFAULT_GGUF_FILE = "Qwen3-8B-Q4_K_M.gguf"   # ~5 GB, runs on CPU or GPU
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_CHUNK_TOKENS = 12000                 # transcript tokens per LLM pass (~30-40 min of talk)
OUTPUT_TOKENS = 6144

SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{SHERPA_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# 3D-Speaker ERes2Net (Apache-2.0); beat WeSpeaker ResNet34 in our multi-speaker tests
EMBEDDING_URL = f"{SHERPA_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

CACHE_DIR = Path(os.environ.get("MEETING_ACTIONS_CACHE", Path.home() / ".cache" / "meeting-actions"))
SAMPLE_RATE = 16000
CHECKPOINT_VERSION = 3  # bump when a step's output changes


def log(msg: str) -> None:
    print(f"[meeting-actions] {msg}", file=sys.stderr, flush=True)


def progress(msg: str) -> None:
    print(f"\r  {msg}\033[K", end="", file=sys.stderr, flush=True)


def progress_done() -> None:
    print(file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Model downloads (always ask first)

ASSUME_YES = False


def human(n: float | None) -> str:
    if not n:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def confirm(what: str, size: int | None) -> None:
    if ASSUME_YES:
        log(f"Downloading {what} ({human(size)}) (--yes)")
        return
    if not sys.stdin.isatty():
        die(f"Model '{what}' must be downloaded first. Re-run interactively or pass --yes.")
    answer = input(f"The model '{what}' ({human(size)}) is not on this machine. Download it now? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        die(f"Cannot continue without '{what}'.")


def url_size(url: str) -> int | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(f"{done * 100 // total:3d}% of {human(total)}")
    progress_done()
    tmp.rename(dest)


def hf_repo_size(repo: str, filename: str | None = None) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        return sum(s.size or 0 for s in info.siblings if filename is None or s.rfilename == filename) or None
    except Exception:
        return None


def ensure_whisper(name: str) -> str:
    """Return a local path to a faster-whisper model, asking before downloading."""
    from faster_whisper.utils import _MODELS, download_model

    if Path(name).is_dir():
        return name
    try:
        return download_model(name, local_files_only=True)
    except Exception:
        repo = _MODELS.get(name, name)
        confirm(f"Whisper {name} ({repo})", hf_repo_size(repo))
        return download_model(name)


def ensure_diarization_models() -> tuple[Path, Path]:
    seg_model = CACHE_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb_model = CACHE_DIR / Path(EMBEDDING_URL).name
    if not seg_model.exists():
        confirm("pyannote segmentation 3.0 (ONNX)", url_size(SEGMENTATION_URL))
        archive = CACHE_DIR / Path(SEGMENTATION_URL).name
        download_url(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(CACHE_DIR, filter="data")
        archive.unlink()
    if not emb_model.exists():
        confirm("3D-Speaker ERes2Net speaker embeddings (ONNX)", url_size(EMBEDDING_URL))
        download_url(EMBEDDING_URL, emb_model)
    return seg_model, emb_model


def ensure_gguf(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    if Path(filename).is_file():
        return filename
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        confirm(f"{repo}/{filename}", hf_repo_size(repo, filename))
        return hf_hub_download(repo, filename)


# --------------------------------------------------------------------------
# Checkpoints: the slow steps (diarization, transcription) are saved next to the
# audio so an interrupted 2-hour run doesn't start over.


def load_checkpoint(path: Path, params: dict):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("params") != params:
        return None
    log(f"Reusing {path.name} (delete it or pass --fresh to redo this step)")
    return data["data"]


def save_checkpoint(path: Path, params: dict, data) -> None:
    path.write_text(json.dumps({"params": params, "data": data}, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Audio


def decode_audio(path: str):
    """Decode any audio/video file to 16 kHz mono float32 (PyAV bundles FFmpeg)."""
    import av
    import numpy as np

    chunks = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            die(f"No audio track in {path}")
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        die(f"Could not decode any audio from {path}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------
# 1. Diarization


def make_diarizer_parts(threads: int):
    import sherpa_onnx

    seg_model, emb_model = ensure_diarization_models()
    seg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
        num_threads=threads,
    )
    emb = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=threads)
    return seg, emb


def diarize(audio, num_speakers: int | None, threshold: float, min_speaker_seconds: float) -> list[list]:
    """Return speaker turns [start_s, end_s, speaker_id], sorted by start."""
    import sherpa_onnx

    threads = max(1, (os.cpu_count() or 2) - 1)
    seg, emb = make_diarizer_parts(threads)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=seg,
        embedding=emb,
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        die("Invalid diarization config (model files missing or corrupt?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    t0 = time.time()

    def cb(done, total):
        pct = done / max(total, 1)
        eta = (time.time() - t0) / pct * (1 - pct) if pct > 0.02 else 0
        progress(f"speakers: {pct:4.0%}" + (f", ~{fmt_time(eta)} left" if eta else ""))
        return 0

    log(f"Identifying speakers in {fmt_time(len(audio) / SAMPLE_RATE)} of audio...")
    turns = [[s.start, s.end, s.speaker] for s in sd.process(audio, callback=cb).sort_by_start_time()]
    progress_done()

    return refine_speakers(audio, turns, emb, num_speakers, min_speaker_seconds)


def refine_speakers(audio, turns: list[list], emb_config, num_speakers: int | None,
                    min_seconds: float) -> list[list]:
    """
    Second pass over sherpa-onnx's speaker turns, using one voice embedding per turn:
    - with --speakers N: re-cluster the turns into exactly N voices (sherpa's own clustering
      sometimes returns fewer than requested);
    - without it: automatic clustering on long recordings tends to invent extra "speakers"
      with only a few seconds of speech, so those tiny clusters are merged into the most
      similar real speaker.
    """
    import numpy as np
    import sherpa_onnx

    if len(turns) < 2:
        return turns
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(emb_config)
    emb: list = []
    for i, (s, e, _) in enumerate(turns):
        piece = audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)]
        if len(piece) < SAMPLE_RATE // 2:  # < 0.5 s: too short for a reliable voice print
            emb.append(None)
            continue
        if len(piece) < SAMPLE_RATE:  # the embedding model wants >= ~1 s
            piece = np.tile(piece, int(np.ceil(SAMPLE_RATE / len(piece))))[:SAMPLE_RATE]
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, piece[: 30 * SAMPLE_RATE])
        stream.input_finished()
        v = np.array(extractor.compute(stream))
        emb.append(v / (np.linalg.norm(v) + 1e-9))
        if i % 20 == 0:
            progress(f"voice prints: {i + 1}/{len(turns)} turns")
    progress_done()

    have = [i for i, v in enumerate(emb) if v is not None]
    if num_speakers and len(have) >= num_speakers:
        from scipy.cluster.hierarchy import fcluster, linkage

        X = np.stack([emb[i] for i in have])
        labels = fcluster(linkage(X, method="average", metric="cosine"), t=num_speakers, criterion="maxclust")
        for i, lab in zip(have, labels):
            turns[i][2] = int(lab)
    elif not num_speakers:
        talk: dict[int, float] = {}
        for s, e, k in turns:
            talk[k] = talk.get(k, 0.0) + e - s
        limit = min(min_seconds, 0.02 * sum(talk.values()))  # 20 s for long meetings, less for short ones
        small = {k for k, v in talk.items() if v < limit}
        big = [k for k in talk if k not in small]
        if small and big:
            def centroid(k):
                vs = [emb[i] for i in have if turns[i][2] == k]
                if not vs:
                    return None
                c = np.mean(vs, axis=0)
                return c / (np.linalg.norm(c) + 1e-9)

            cents = {k: c for k in big if (c := centroid(k)) is not None}
            for i in have:
                if turns[i][2] in small and cents:
                    turns[i][2] = max(cents, key=lambda b: float(emb[i] @ cents[b]))
            log(f"Merged {len(small)} tiny speaker cluster(s) (< {limit:.0f}s of speech) into similar voices")

    # Turns without a voice print (or still in a tiny cluster) take the label of the nearest turn.
    valid = {t[2] for i, t in enumerate(turns) if emb[i] is not None}
    for i, t in enumerate(turns):
        if emb[i] is None or t[2] not in valid:
            near = [u for j, u in enumerate(turns) if emb[j] is not None and abs(j - i) <= 4]
            if near:
                t[2] = min(near, key=lambda u: abs(u[0] - t[0]))[2]
    log(f"Speakers found: {len({t[2] for t in turns})}")
    return turns


# --------------------------------------------------------------------------
# 2-3. Language detection + transcription


@dataclass
class Word:
    start: float
    end: float
    text: str
    lang: str
    speaker: str = "?"


def speech_pieces(audio, turns: list[list]) -> list[tuple[float, float]]:
    """Speech regions (Silero VAD, <= 25 s), further split wherever the speaker changes."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    vad = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=25, min_silence_duration_ms=300,
                                                  speech_pad_ms=200))
    changes = []
    for a, b in zip(turns, turns[1:]):
        if a[2] != b[2]:
            changes.append((a[1] + b[0]) / 2 if b[0] > a[1] else b[0])

    pieces = []
    for v in vad:
        s, e = v["start"] / SAMPLE_RATE, v["end"] / SAMPLE_RATE
        cuts = [c for c in changes if s + 0.5 < c < e - 0.5]
        bounds = [s, *cuts, e]
        pieces += list(zip(bounds, bounds[1:]))
    return pieces


def detect_languages(audio, pieces, allowed: list[str] | None, lid_name: str, device: str) -> list[str]:
    """Detect the spoken language of each piece with a small Whisper model."""
    from faster_whisper import WhisperModel

    if allowed and len(allowed) == 1:
        return [allowed[0]] * len(pieces)

    lid = WhisperModel(ensure_whisper(lid_name), device=device, compute_type=compute_type(device))
    probs: list[dict[str, float] | None] = []
    t0 = time.time()
    for i, (s, e) in enumerate(pieces):
        if e - s < 1.0:  # too short to judge; inherit from neighbours below
            probs.append(None)
            continue
        _, _, all_probs = lid.detect_language(audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)])
        probs.append(dict(all_probs))
        if i % 10 == 0:
            progress(f"languages: {i + 1}/{len(pieces)} pieces ({time.time() - t0:.0f}s)")
    progress_done()

    if not allowed:
        # Auto: keep every language that covers >= 10% of the detected speech.
        share: dict[str, float] = {}
        for (s, e), p in zip(pieces, probs):
            if p:
                top = max(p, key=p.get)
                share[top] = share.get(top, 0.0) + e - s
        total = sum(share.values()) or 1
        allowed = [l for l, d in sorted(share.items(), key=lambda x: -x[1]) if d / total >= 0.10] or ["en"]
        log(f"Detected languages: {', '.join(allowed)}  (set --languages to override)")

    langs: list[str | None] = [max(allowed, key=lambda l: p.get(l, 0.0)) if p else None for p in probs]
    # Fill very short pieces from the nearest neighbour that has a language.
    for i, l in enumerate(langs):
        if l is None:
            near = [langs[j] for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(langs) and langs[j]]
            langs[i] = near[0] if near else allowed[0]
    counts = {l: sum(1 for x in langs if x == l) for l in allowed}
    log("Pieces per language: " + ", ".join(f"{l}={n}" for l, n in counts.items()))
    return langs


def group_chunks(pieces, langs, max_len: float = 28.0) -> list[tuple[float, float, str]]:
    """Merge consecutive same-language pieces into chunks of at most max_len seconds."""
    chunks: list[list] = []
    for (s, e), lang in zip(pieces, langs):
        if chunks and chunks[-1][2] == lang and e - chunks[-1][0] <= max_len and s - chunks[-1][1] < 3.0:
            chunks[-1][1] = e
        else:
            chunks.append([s, e, lang])
    return [tuple(c) for c in chunks]


def compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


LANG_NAMES = {"ar": "Arabic", "en": "English", "fr": "French", "de": "German", "es": "Spanish",
              "tr": "Turkish", "ur": "Urdu", "fa": "Persian", "hi": "Hindi"}


def transcribe(audio, chunks, model_name: str, device: str, batch_size: int) -> list[Word]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    log(f"Loading Whisper '{model_name}' on {device}...")
    model = WhisperModel(ensure_whisper(model_name), device=device, compute_type=compute_type(device))
    pipe = BatchedInferencePipeline(model)
    total_speech = sum(e - s for s, e, _ in chunks) or 1
    done_speech, t0 = 0.0, time.time()

    words: list[Word] = []
    for lang in dict.fromkeys(l for _, _, l in chunks):
        clips = [{"start": s, "end": e} for s, e, l in chunks if l == lang]
        segments, _ = pipe.transcribe(audio, language=lang, clip_timestamps=clips, word_timestamps=True,
                                      batch_size=batch_size, without_timestamps=False)
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(w.start, w.end, w.word, lang))
            done_speech += seg.end - seg.start
            rate = done_speech / max(time.time() - t0, 1e-6)
            left = (total_speech - done_speech) / rate if rate else 0
            progress(f"transcribing {LANG_NAMES.get(lang, lang)}: {min(done_speech / total_speech, 1):4.0%} "
                     f"of speech, {rate:.1f}x real time, ~{fmt_time(max(left, 0))} left")
    progress_done()
    log(f"Transcribed {len(words):,} words in {fmt_time(time.time() - t0)}.")
    words.sort(key=lambda w: w.start)
    return words


# --------------------------------------------------------------------------
# 4. Alignment


def speaker_letter(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"S{i + 1}"


def assign_speakers(words: list[Word], turns: list[list]) -> None:
    """Give each word the speaker whose turn overlaps it most (or the nearest turn)."""
    import numpy as np

    if not turns:
        for w in words:
            w.speaker = "A"
        return
    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    ids = [t[2] for t in turns]

    letters: dict[int, str] = {}  # A, B, C... in order of first appearance
    for spk in ids:
        letters.setdefault(spk, speaker_letter(len(letters)))

    for w in words:
        overlap = np.minimum(ends, w.end) - np.maximum(starts, w.start)
        i = int(overlap.argmax())
        if overlap[i] <= 0:  # the word falls in a gap: use the closest turn
            i = int(np.maximum(starts - w.end, w.start - ends).argmin())
        w.speaker = letters[ids[i]]


def build_utterances(words: list[Word], max_gap: float = 2.0, soft_max: float = 30.0) -> list[dict]:
    """Group consecutive words of the same speaker (and language) into utterances."""
    utterances: list[dict] = []
    cur: list[Word] = []

    def flush():
        text = re.sub(r"\s+", " ", "".join(w.text for w in cur)).strip()
        if text:
            utterances.append({"id": len(utterances) + 1, "speaker": cur[0].speaker, "lang": cur[0].lang,
                               "start_ms": int(cur[0].start * 1000), "end_ms": int(cur[-1].end * 1000),
                               "text": text})
        cur.clear()

    for w in words:
        if cur and (w.speaker != cur[-1].speaker or w.lang != cur[-1].lang
                    or w.start - cur[-1].end > max_gap
                    or (cur[-1].end - cur[0].start > soft_max and cur[-1].text.rstrip().endswith((".", "?", "!", "؟")))):
            flush()
        cur.append(w)
    if cur:
        flush()
    return utterances


# --------------------------------------------------------------------------
# 5. Analysis with a local LLM

SYSTEM_PROMPT = """You analyse meeting transcripts and turn them into clear, actionable minutes.

The transcript was produced by automatic speech recognition with speaker diarization.
Each line is: #<utterance number> [<timestamp>] Speaker <label>: <text>.
The meeting may mix languages (for example Arabic and English, sometimes within one
sentence). Expect recognition errors and occasional mis-attributed speakers; interpret
them sensibly from context, and never invent content that was not said.

Produce:
- speakers: for each speaker label, the person's name if the transcript reveals it, else empty.
  Typical clues: "Sara, can you..." -> the NEXT speaker is usually Sara; "Thanks, Omar" ->
  usually the PREVIOUS speaker; "Sure, Nizar" -> the person just addressed is the previous
  speaker; self-introductions. Write the name exactly as it is written in the transcript,
  and never make one up. role: only if it is clear from what was said, otherwise empty.
- discussion_points: the distinct topics in the order discussed, each with a concise
  summary of what was said and by whom (use names where known).
- decisions: things the group actually agreed or settled.
- action_items: EVERY concrete follow-up: anything someone offered or committed to
  ("I can...", "I will...", "I'll check...", "سأرسل", "سأتواصل") and anything someone was asked to do.
  owner is the person who must DO the task: the speaker who volunteered, or the person who
  was asked ("Sara, please send..." -> Sara, not the one asking). Give the owner as a speaker
  label (e.g. "B") if you know which label that person is, otherwise their name; empty if
  nobody was assigned. due is the deadline as stated ("by Wednesday", "today"), or empty.
- open_questions: issues raised but left unresolved.

For every item, list the utterance numbers (the #N values) of the lines where that item is
actually said, most relevant first. Do not cite lines that merely open the meeting or are
unrelated. Use only numbers that appear in the transcript."""


def _item(props: dict) -> dict:
    props = {**props, "utterance_ids": {"type": "array", "items": {"type": "integer"}}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
_PRIORITY = {"type": "string", "enum": ["high", "medium", "low"]}
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR,
        "summary": _STR,
        "speakers": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": _STR, "inferred_name": _STR, "role": _STR},
            "required": ["label", "inferred_name", "role"],
            "additionalProperties": False,
        }},
        "discussion_points": {"type": "array", "items": _item({"topic": _STR, "summary": _STR})},
        "decisions": {"type": "array", "items": _item({"decision": _STR})},
        "action_items": {"type": "array", "items": _item({
            "task": _STR, "owner": _STR, "due": _STR, "priority": _PRIORITY})},
        "open_questions": {"type": "array", "items": _item({"question": _STR})},
    },
    "required": ["title", "summary", "speakers", "discussion_points", "decisions", "action_items", "open_questions"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"title": _STR, "summary": _STR},
    "required": ["title", "summary"],
    "additionalProperties": False,
}

# For merging items found in different parts of a long meeting.
MERGE_FIELDS = {
    "action_items": ("task", {"task": _STR, "owner": _STR, "due": _STR, "priority": _PRIORITY}),
    "decisions": ("decision", {"decision": _STR}),
    "open_questions": ("question", {"question": _STR}),
}


class LocalLLM:
    """Small wrapper so llama.cpp and Ollama look the same to the rest of the script."""

    def __init__(self, args):
        self.backend = args.llm
        self.n_ctx = args.ctx
        if self.backend == "llamacpp":
            try:
                from llama_cpp import Llama
            except ImportError:
                die("llama-cpp-python is not installed: pip install llama-cpp-python")
            path = ensure_gguf(args.gguf_repo, args.gguf_file)
            log(f"Loading {Path(path).name} (context {self.n_ctx} tokens)...")
            self.llm = Llama(model_path=path, n_ctx=self.n_ctx, n_gpu_layers=args.gpu_layers, verbose=False)
        else:
            self.url = args.ollama_url.rstrip("/")
            self.model = args.ollama_model
            self._ensure_ollama_model()

    # -- Ollama helpers
    def _ollama(self, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=7200)
        except Exception as e:
            die(f"Cannot reach Ollama at {self.url} ({e}). Is `ollama serve` running?")

    def _ensure_ollama_model(self) -> None:
        with self._ollama("/api/tags") as r:
            names = {m["name"] for m in json.loads(r.read()).get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return
        confirm(f"Ollama {self.model}", None)
        with self._ollama("/api/pull", {"model": self.model, "stream": True}) as r:
            for line in r:
                ev = json.loads(line)
                if ev.get("error"):
                    die(f"Ollama pull failed: {ev['error']}")
                if ev.get("total"):
                    progress(f"{ev.get('status', '')} {ev.get('completed', 0) * 100 // ev['total']:3d}%")
        progress_done()

    # -- common API
    def count_tokens(self, text: str) -> int:
        if self.backend == "llamacpp":
            return len(self.llm.tokenize(text.encode(), add_bos=False))
        return len(text) // 3  # rough estimate; Ollama has no tokenize endpoint

    def json_chat(self, system: str, user: str, schema: dict, max_tokens: int, label: str) -> dict:
        # "/no_think" turns off Qwen3's reasoning preamble; other models ignore it.
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user + "\n/no_think"}]
        t0 = time.time()
        progress(f"{label}: reading {self.count_tokens(user):,} tokens...")
        parts, finish, n = [], None, 0
        if self.backend == "llamacpp":
            stream = self.llm.create_chat_completion(
                messages=messages, response_format={"type": "json_object", "schema": schema},
                temperature=0.1, max_tokens=max_tokens, stream=True)
            for ev in stream:
                choice = ev["choices"][0]
                if piece := choice["delta"].get("content"):
                    parts.append(piece)
                    n += 1
                    progress(f"{label}: writing, {n} tokens ({time.time() - t0:.0f}s)")
                finish = choice.get("finish_reason") or finish
        else:
            body = {"model": self.model, "messages": messages, "format": schema, "stream": True,
                    "options": {"num_ctx": self.n_ctx, "temperature": 0.1, "num_predict": max_tokens}}
            with self._ollama("/api/chat", body) as r:
                for line in r:
                    ev = json.loads(line)
                    if piece := ev.get("message", {}).get("content"):
                        parts.append(piece)
                        n += 1
                        progress(f"{label}: writing, {n} tokens ({time.time() - t0:.0f}s)")
                    if ev.get("done"):
                        finish = ev.get("done_reason")
        progress(f"{label}: done in {time.time() - t0:.0f}s")
        progress_done()
        try:
            return json.loads("".join(parts))
        except json.JSONDecodeError:
            hint = " (it hit the output limit; try a smaller --chunk-tokens)" if finish == "length" else ""
            die(f"The model did not return valid JSON{hint}.")


def transcript_lines(utterances: list[dict]) -> list[str]:
    return [f"#{u['id']} [{fmt_time(u['start_ms'])}] Speaker {u['speaker']}: {u['text']}" for u in utterances]


def split_for_llm(utterances: list[dict], llm: LocalLLM, budget: int) -> list[list[dict]]:
    """Split the transcript into parts that fit the context, cutting at the longest pause."""
    parts, cur, cur_tokens = [], [], 0
    for u, line in zip(utterances, transcript_lines(utterances)):
        t = llm.count_tokens(line) + 1
        if cur and cur_tokens + t > budget:
            # cut at the longest pause in the last 20% of the part
            tail = max(1, len(cur) // 5)
            gaps = [(cur[i]["start_ms"] - cur[i - 1]["end_ms"], i) for i in range(len(cur) - tail, len(cur))]
            _, cut = max(gaps)
            parts.append(cur[:cut])
            cur = cur[cut:]
            cur_tokens = sum(llm.count_tokens(x) + 1 for x in transcript_lines(cur))
        cur.append(u)
        cur_tokens += t
    if cur:
        parts.append(cur)
    return parts


def analyse(utterances: list[dict], llm: LocalLLM, known_names: dict[str, str], minutes_lang: str,
            chunk_tokens: int) -> dict:
    system = (SYSTEM_PROMPT + f"\n\nWrite titles, summaries, topics and items in {minutes_lang}. "
              "Keep people's names as written in the transcript.")
    names_hint = ""
    if known_names:
        names_hint = "\nKnown speaker identities (authoritative): " + "; ".join(
            f"Speaker {k} = {v}" for k, v in known_names.items()) + "\n"

    parts = split_for_llm(utterances, llm, chunk_tokens)
    if len(parts) > 1:
        log(f"Long meeting: analysing it in {len(parts)} parts, then merging.")
    results = []
    for i, part in enumerate(parts, 1):
        note = f" This is part {i} of {len(parts)} of a longer meeting; analyse only this part." if len(parts) > 1 else ""
        label = f"LLM part {i}/{len(parts)} [{fmt_time(part[0]['start_ms'])}-{fmt_time(part[-1]['end_ms'])}]"
        results.append(llm.json_chat(
            system,
            "<transcript>\n" + "\n".join(transcript_lines(part)) + f"\n</transcript>\n{names_hint}\n"
            f"Produce the meeting minutes.{note}",
            ANALYSIS_SCHEMA, OUTPUT_TOKENS, label))

    return results[0] if len(results) == 1 else merge_parts(results, llm, minutes_lang)


def merge_parts(results: list[dict], llm: LocalLLM, minutes_lang: str) -> dict:
    merged: dict = {"discussion_points": [p for r in results for p in r.get("discussion_points", [])]}

    speakers: dict[str, dict] = {}
    for r in results:
        for s in r.get("speakers", []):
            prev = speakers.get(s.get("label", ""))
            if prev is None or (not prev.get("inferred_name") and s.get("inferred_name")):
                speakers[s.get("label", "")] = s
    merged["speakers"] = list(speakers.values())

    # Items about the same thing often come up in several parts: let the LLM group duplicates,
    # then union their citations here (so no citation is lost or invented).
    for section, (key, fields) in MERGE_FIELDS.items():
        items = [it for r in results for it in r.get(section, [])]
        if len(items) < 2:
            merged[section] = items
            continue
        listing = "\n".join(
            f"[{i}] " + " | ".join(f"{f}: {it.get(f, '')}" for f in fields) for i, it in enumerate(items))
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {"merged_from": {"type": "array", "items": {"type": "integer"}}, **fields},
            "required": ["merged_from", *fields], "additionalProperties": False}}},
            "required": ["items"], "additionalProperties": False}
        out = llm.json_chat(
            f"You merge duplicate {section.replace('_', ' ')} collected from consecutive parts of one meeting. "
            "Combine entries that refer to the same thing into one (keep the most specific owner and deadline); "
            f"keep distinct entries separate. Write in {minutes_lang}. List in merged_from the [numbers] of the "
            "entries each output item combines. Every input number must appear exactly once.",
            listing, schema, OUTPUT_TOKENS, f"LLM merging {section.replace('_', ' ')}")
        used, final = set(), []
        for g in out.get("items", []):
            ids = [i for i in g.get("merged_from", []) if isinstance(i, int) and 0 <= i < len(items) and i not in used]
            if not ids:
                continue
            used.update(ids)
            final.append({**{f: g.get(f, "") for f in fields},
                          "utterance_ids": [u for i in ids for u in items[i].get("utterance_ids", [])]})
        final += [items[i] for i in range(len(items)) if i not in used]  # never drop an item
        merged[section] = final

    parts_text = "\n\n".join(f"Part {i}: {r.get('summary', '')}" for i, r in enumerate(results, 1))
    top = llm.json_chat(
        f"You write a title and a short overall summary (in {minutes_lang}) of a meeting, "
        "given summaries of its consecutive parts.",
        parts_text, SUMMARY_SCHEMA, 1024, "LLM overall summary")
    merged.update(title=top.get("title", "Meeting"), summary=top.get("summary", ""))
    return merged


# --------------------------------------------------------------------------
# 6. Resolve citations to timestamps + render


def attach_timestamps(a: dict, utterances: list[dict], known_names: dict[str, str]) -> dict:
    by_id = {u["id"]: u for u in utterances}
    # Small local models sometimes invent names: accept a name only if it occurs in the transcript.
    spoken = " ".join(u["text"] for u in utterances).lower()

    names: dict[str, str] = {}
    for s in a.get("speakers", []):
        label = str(s.get("label", "")).upper().removeprefix("SPEAKER ").strip()
        s["label"] = label
        guess = str(s.get("inferred_name") or "").strip()
        if guess and not all(part.lower() in spoken for part in guess.split()):
            guess = ""
        s["inferred_name"] = guess
        names[label] = known_names.get(label) or guess or f"Speaker {label}"
    for label in sorted({u["speaker"] for u in utterances}):  # speakers the LLM skipped
        if label not in names:
            names[label] = known_names.get(label, f"Speaker {label}")
            a.setdefault("speakers", []).append({"label": label, "inferred_name": "", "role": ""})
    a["speakers"] = [s for s in a["speakers"] if s["label"] in {u["speaker"] for u in utterances}]

    def who(value: str) -> str:
        """Speaker label -> display name; a plain name is passed through."""
        value = str(value or "").strip()
        label = value.upper().removeprefix("SPEAKER ").strip()
        return names.get(label, value)

    def refs(ids) -> list[dict]:
        out, seen = [], set()
        for i in ids or []:
            if isinstance(i, int) and i in by_id and i not in seen:  # drop invented citations
                seen.add(i)
                u = by_id[i]
                out.append({"utterance": i, "time": fmt_time(u["start_ms"]), "start_ms": u["start_ms"],
                            "end_ms": u["end_ms"], "speaker": who(u["speaker"]), "quote": u["text"]})
        return sorted(out, key=lambda x: x["start_ms"])

    for section in ("discussion_points", "decisions", "action_items", "open_questions"):
        for entry in a.setdefault(section, []):
            entry["references"] = refs(entry.pop("utterance_ids", []))
    for item in a["action_items"]:
        item["owner"] = who(item.get("owner", ""))
    for s in a["speakers"]:
        s["name"] = names[s["label"]]
        talk = sum(u["end_ms"] - u["start_ms"] for u in utterances if u["speaker"] == s["label"])
        s["speaking_time"] = fmt_time(talk)

    a["duration"] = fmt_time(utterances[-1]["end_ms"])
    return a


def render_markdown(r: dict) -> str:
    def times(refs):
        return ", ".join(f"`{x['time']}`" for x in refs) or "_no reference_"

    def cell(s):
        return str(s or "—").replace("|", "\\|").replace("\n", " ")

    people = ", ".join(f"{s['name']}" + (f" ({s['role']})" if s.get("role") else "") + f" – {s['speaking_time']}"
                       for s in r["speakers"])
    md = [f"# {r.get('title', 'Meeting')}", "",
          f"**Duration:** {r['duration']}  ", f"**Participants (speaking time):** {people}", "",
          "## Summary", "", r.get("summary", ""), "", "## Action items", ""]

    if r["action_items"]:
        md += ["| # | Action | Owner | Due | Priority | Timestamps |", "|---|---|---|---|---|---|"]
        for i, ai in enumerate(r["action_items"], 1):
            md.append(f"| {i} | {cell(ai.get('task'))} | {cell(ai.get('owner'))} | {cell(ai.get('due'))} "
                      f"| {ai.get('priority', '')} | {times(ai['references'])} |")
    else:
        md.append("_None recorded._")

    md += ["", "## Decisions", ""]
    md += [f"- {d.get('decision', '')} — {times(d['references'])}" for d in r["decisions"]] or ["_None recorded._"]

    md += ["", "## Discussion points", ""]
    for i, p in enumerate(r["discussion_points"], 1):
        md += [f"### {i}. {p.get('topic', '')}  ({times(p['references'])})", "", p.get("summary", ""), ""]
        quotes = [f"> `{q['time']}` **{q['speaker']}:** {q['quote']}" for q in p["references"][:3]]
        if quotes:
            md += ["\n>\n".join(quotes), ""]

    if r["open_questions"]:
        md += ["## Open questions", ""]
        md += [f"- {q.get('question', '')} — {times(q['references'])}" for q in r["open_questions"]]
        md.append("")
    return "\n".join(md).rstrip() + "\n"


# --------------------------------------------------------------------------
# CLI


def fmt_time(t: float | int) -> str:
    """Seconds (float) or milliseconds (int) -> HH:MM:SS."""
    s = int(t // 1000) if isinstance(t, int) else int(t)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_names(v: str | None) -> dict[str, str]:
    out = {}
    for pair in filter(None, (p.strip() for p in (v or "").split(","))):
        label, _, name = pair.partition("=")
        if label.strip() and name.strip():
            out[label.strip().upper()] = name.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Local meeting audio -> timestamped action points.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("audio", nargs="?", help="meeting recording (mp3, m4a, wav, mp4, ...)")
    p.add_argument("--transcript", help="skip transcription and analyse an existing *.transcript.json")
    p.add_argument("--languages", help='spoken languages, e.g. "ar,en" (default: detect automatically)')
    p.add_argument("--speakers", type=int, help="number of speakers, if known (improves speaker separation)")
    p.add_argument("--names", help='map speaker labels to names, e.g. "A=Nizar,B=Sara"')
    p.add_argument("--minutes-language", default="English", help="language of the written minutes (default English)")
    p.add_argument("--out", help="output path prefix (default: next to the input)")
    p.add_argument("--yes", "-y", action="store_true", help="download missing models without asking")
    p.add_argument("--fresh", action="store_true", help="ignore saved checkpoints and redo every step")
    p.add_argument("--no-analysis", action="store_true", help="only produce the diarized transcript")

    g = p.add_argument_group("speech")
    g.add_argument("--whisper", default=DEFAULT_WHISPER, help=f"faster-whisper model (default {DEFAULT_WHISPER})")
    g.add_argument("--lid-model", default=DEFAULT_LID_MODEL, help="Whisper model used to detect languages")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--batch-size", type=int, default=8, help="Whisper batch size (lower it if you run out of memory)")
    g.add_argument("--cluster-threshold", type=float, default=0.7,
                   help="speaker clustering threshold without --speakers; lower = more speakers (default 0.7)")
    g.add_argument("--min-speaker-seconds", type=float, default=20.0,
                   help="without --speakers, speakers with less speech than this (or 2%% of the meeting, "
                        "whichever is smaller) are merged into the closest voice (default 20)")

    g = p.add_argument_group("LLM")
    g.add_argument("--llm", default="llamacpp", choices=["llamacpp", "ollama"])
    g.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO, help="Hugging Face repo with the GGUF model")
    g.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE, help="GGUF file name in that repo, or a local path")
    g.add_argument("--gpu-layers", type=int, default=-1, help="layers to offload to GPU (-1 = all, 0 = CPU only)")
    g.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    g.add_argument("--ollama-url", default="http://localhost:11434")
    g.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS,
                   help=f"transcript tokens per LLM pass; longer meetings are split (default {DEFAULT_CHUNK_TOKENS})")
    args = p.parse_args()
    args.ctx = args.chunk_tokens + OUTPUT_TOKENS + 2048

    global ASSUME_YES
    ASSUME_YES = args.yes
    known_names = parse_names(args.names)
    languages = [l.strip().lower() for l in args.languages.split(",") if l.strip()] if args.languages else None

    if args.transcript:
        utterances = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        base = re.sub(r"(\.transcript)?\.json$", "", args.transcript)
    elif args.audio:
        if not Path(args.audio).is_file():
            die(f"Audio file not found: {args.audio}")
        base = args.audio
        if args.whisper in ("tiny", "base", "small") or args.whisper.endswith(".en"):
            if languages != ["en"]:
                log(f"Note: Whisper '{args.whisper}' is weak outside English; use large-v3-turbo for Arabic.")

        # Ask about every missing model up front, so a long run isn't interrupted halfway.
        ensure_diarization_models()
        if languages is None or len(languages) > 1:
            ensure_whisper(args.lid_model)
        ensure_whisper(args.whisper)
        if not args.no_analysis and args.llm == "llamacpp":
            ensure_gguf(args.gguf_repo, args.gguf_file)

        device = resolve_device(args.device)
        log("Decoding audio...")
        audio = decode_audio(args.audio)
        log(f"Audio: {fmt_time(len(audio) / SAMPLE_RATE)}. Whisper and the LLM run on {device}.")

        stat = Path(args.audio).stat()
        source = {"size": stat.st_size, "mtime": int(stat.st_mtime), "v": CHECKPOINT_VERSION}

        turns_params = {**source, "speakers": args.speakers, "threshold": args.cluster_threshold,
                        "min_speaker_seconds": args.min_speaker_seconds}
        turns_ckpt = Path(f"{base}.speakers.ckpt.json")
        turns = None if args.fresh else load_checkpoint(turns_ckpt, turns_params)
        if turns is None:
            turns = diarize(audio, args.speakers, args.cluster_threshold, args.min_speaker_seconds)
            save_checkpoint(turns_ckpt, turns_params, turns)

        words_params = {**turns_params, "languages": languages, "whisper": args.whisper, "lid": args.lid_model}
        words_ckpt = Path(f"{base}.words.ckpt.json")
        saved = None if args.fresh else load_checkpoint(words_ckpt, words_params)
        if saved is None:
            pieces = speech_pieces(audio, turns)
            log(f"Detecting the language of {len(pieces)} speech pieces...")
            langs = detect_languages(audio, pieces, languages, args.lid_model, device)
            chunks = group_chunks(pieces, langs)
            words = transcribe(audio, chunks, args.whisper, device, args.batch_size)
            save_checkpoint(words_ckpt, words_params, [asdict(w) for w in words])
        else:
            words = [Word(**w) for w in saved]

        assign_speakers(words, turns)
        utterances = build_utterances(words)
        Path(f"{base}.transcript.json").write_text(json.dumps(utterances, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(f"{base}.transcript.txt").write_text("\n".join(transcript_lines(utterances)) + "\n", encoding="utf-8")
        log(f"Transcript: {len(utterances)} utterances, {len({u['speaker'] for u in utterances})} speakers "
            f"-> {base}.transcript.txt")
    else:
        p.print_help()
        sys.exit(1)

    if not utterances:
        die("The transcript is empty - nothing to analyse.")
    if args.no_analysis:
        return

    llm = LocalLLM(args)
    analysis = analyse(utterances, llm, known_names, args.minutes_language, args.chunk_tokens)
    report = attach_timestamps(analysis, utterances, known_names)

    out = args.out or base
    Path(f"{out}.actions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(f"{out}.actions.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"Wrote {out}.actions.md and {out}.actions.json")


if __name__ == "__main__":
    main()
