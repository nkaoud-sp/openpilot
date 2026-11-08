#!/usr/bin/env python3
"""
meeting_actions.py - turn a recorded meeting into speaker-attributed,
timestamped minutes and action points. Everything runs locally with
open-source models; nothing is sent to a cloud service.

Pipeline
  1. Speech-to-text with word timestamps ... faster-whisper (Whisper, MIT)
  2. Who-spoke-when (diarization) .......... sherpa-onnx (pyannote-3.0 segmentation
                                             + 3D-Speaker ERes2Net embeddings)
  3. Words + speakers -> utterances ........ this script
  4. Topics / decisions / action items ..... local LLM: llama.cpp (default, Qwen3-8B GGUF)
                                             or Ollama
  5. LLM cites utterance numbers; this script maps them back to real
     timestamps, so timestamps are never invented by the model.

Models are downloaded on first use, after asking you (use --yes to skip the question).

Examples
  python meeting_actions.py meeting.mp3
  python meeting_actions.py meeting.m4a --speakers 3 --names "A=Nizar,B=Sara"
  python meeting_actions.py --transcript meeting.transcript.json      # re-run analysis only
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
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults (all open-source, ungated)

DEFAULT_WHISPER = "large-v3-turbo"          # try "small" or "medium" for faster CPU runs
DEFAULT_GGUF_REPO = "Qwen/Qwen3-8B-GGUF"     # Apache-2.0
DEFAULT_GGUF_FILE = "Qwen3-8B-Q4_K_M.gguf"   # ~5 GB, runs on CPU or GPU
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_CTX = 32768

SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{SHERPA_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# 3D-Speaker ERes2Net (Apache-2.0); beat WeSpeaker ResNet34 in our multi-speaker tests
EMBEDDING_URL = f"{SHERPA_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

CACHE_DIR = Path(os.environ.get("MEETING_ACTIONS_CACHE", Path.home() / ".cache" / "meeting-actions"))
SAMPLE_RATE = 16000


def log(msg: str) -> None:
    print(f"[meeting-actions] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Model downloads (always ask first)

ASSUME_YES = False


def human(n: int | None) -> str:
    if not n:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def confirm(what: str, size: int | None) -> None:
    question = f"The model '{what}' ({human(size)}) is not on this machine. Download it now? [y/N] "
    if ASSUME_YES:
        log(f"Downloading {what} ({human(size)}) (--yes)")
        return
    if not sys.stdin.isatty():
        die(f"Model '{what}' must be downloaded first. Re-run interactively or pass --yes.")
    if input(question).strip().lower() not in ("y", "yes"):
        die(f"Cannot continue without '{what}'.")


def url_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d}% of {human(total)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    tmp.rename(dest)


def hf_repo_size(repo: str, filename: str | None = None) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        files = [s for s in info.siblings if filename is None or s.rfilename == filename]
        return sum(s.size or 0 for s in files) or None
    except Exception:
        return None


def ensure_whisper(name: str) -> str:
    """Return a local path to the faster-whisper model, asking before downloading."""
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
    seg_dir = CACHE_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_model = seg_dir / "model.onnx"
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
# 1-3. Transcription, diarization, alignment


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str = "?"


def decode_audio(path: str):
    """Decode any audio/video file to 16 kHz mono float32 (PyAV ships its own FFmpeg)."""
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


def transcribe(audio, model_name: str, language: str | None, device: str) -> list[Word]:
    from faster_whisper import WhisperModel

    path = ensure_whisper(model_name)
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    log(f"Loading Whisper '{model_name}' on {device}...")
    model = WhisperModel(path, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # reduces repetition loops on long audio
    )
    log(f"Language: {info.language} ({info.language_probability:.0%}). Transcribing {fmt_time(info.duration)}...")
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(w.start, w.end, w.word))
        print(f"\r  transcribed {fmt_time(seg.end)} / {fmt_time(info.duration)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return words


def diarize(audio, num_speakers: int | None, threshold: float) -> list[tuple[float, float, int]]:
    import sherpa_onnx

    seg_model, emb_model = ensure_diarization_models()
    threads = max(1, (os.cpu_count() or 2) - 1)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        die("Invalid diarization config (model files missing or corrupt?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def progress(done, total):
        print(f"\r  diarization {done * 100 // max(total, 1):3d}%", end="", file=sys.stderr)
        return 0

    log("Identifying speakers...")
    result = sd.process(audio, callback=progress).sort_by_start_time()
    print(file=sys.stderr)
    return [(s.start, s.end, s.speaker) for s in result]


def assign_speakers(words: list[Word], turns: list[tuple[float, float, int]]) -> None:
    """Give each word the speaker whose turn overlaps it most (or the nearest turn)."""
    import numpy as np

    if not turns:
        for w in words:
            w.speaker = "A"
        return

    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    ids = [t[2] for t in turns]

    # Speaker letters in order of first appearance: A, B, C...
    letters: dict[int, str] = {}
    for spk in ids:
        if spk not in letters:
            letters[spk] = speaker_letter(len(letters))

    for w in words:
        overlap = np.minimum(ends, w.end) - np.maximum(starts, w.start)
        i = int(overlap.argmax())
        if overlap[i] <= 0:  # word falls in a gap: use the closest turn
            dist = np.maximum(starts - w.end, w.start - ends)
            i = int(dist.argmin())
        w.speaker = letters[ids[i]]


def speaker_letter(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"S{i + 1}"


def build_utterances(words: list[Word], max_gap: float = 2.0, soft_max: float = 30.0) -> list[dict]:
    """Group consecutive words by the same speaker into utterances."""
    utterances: list[dict] = []
    cur: list[Word] = []

    def flush():
        if cur:
            text = "".join(w.text for w in cur).strip()
            if text:
                utterances.append({
                    "id": len(utterances) + 1,
                    "speaker": cur[0].speaker,
                    "start_ms": int(cur[0].start * 1000),
                    "end_ms": int(cur[-1].end * 1000),
                    "text": re.sub(r"\s+", " ", text),
                })
            cur.clear()

    for w in words:
        if cur:
            new_speaker = w.speaker != cur[-1].speaker
            long_gap = w.start - cur[-1].end > max_gap
            long_and_sentence_end = (
                cur[-1].end - cur[0].start > soft_max and cur[-1].text.rstrip().endswith((".", "?", "!"))
            )
            if new_speaker or long_gap or long_and_sentence_end:
                flush()
        cur.append(w)
    flush()
    return utterances


# --------------------------------------------------------------------------
# 4. Analysis with a local LLM

SYSTEM_PROMPT = """You analyse meeting transcripts and turn them into clear, actionable minutes.

The transcript was produced by automatic speech recognition with speaker
diarization. Each line is: #<utterance number> [<timestamp>] Speaker <label>: <text>.
Expect recognition errors and occasional mis-attributed speakers; interpret them
sensibly from context, and never invent content that was not said.

Produce:
- speakers: for each speaker label, the person's name if the transcript reveals it, else empty.
  Typical clues: "Sara, can you..." -> the NEXT speaker is usually Sara; "Thanks, Omar" ->
  usually the PREVIOUS speaker; "Sure, Nizar" -> the person just addressed is the previous
  speaker; self-introductions. Use only names that appear in the transcript; never make one up.
  role: only if it is clear from what was said, otherwise empty.
- discussion_points: the distinct topics in the order discussed, each with a concise
  summary of what was said and by whom (use names where known).
- decisions: things the group actually agreed or settled.
- action_items: EVERY concrete follow-up: anything someone offered or committed to
  ("I can...", "I will...", "I'll check...") and anything someone was asked to do.
  owner is the person who must DO the task: the speaker who volunteered, or the person who
  was asked ("Sara, please send..." -> Sara, not the one asking). Give the owner as a speaker
  label (e.g. "B") if you know which label that person is, otherwise their name; empty if
  nobody was assigned. due is the deadline exactly as stated ("by Wednesday", "today"), or empty.
- open_questions: issues raised but left unresolved.

For every item, list the utterance numbers (the #N values) of the lines where that item is
actually said, most relevant first. Do not cite lines that merely open the meeting or are
unrelated. Use only numbers that appear in the transcript. Write in the language of the meeting."""


def _item(props: dict) -> dict:
    props = {**props, "utterance_ids": {"type": "array", "items": {"type": "integer"}}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
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
            "task": _STR, "owner": _STR, "due": _STR,
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        })},
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
    def _ollama(self, path: str, body: dict | None = None, stream: bool = False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except Exception as e:
            die(f"Cannot reach Ollama at {self.url} ({e}). Is `ollama serve` running?")
        if stream:
            return r
        with r:
            return json.loads(r.read())

    def _ensure_ollama_model(self) -> None:
        names = {m["name"] for m in self._ollama("/api/tags").get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return
        confirm(f"Ollama {self.model}", None)
        with self._ollama("/api/pull", {"model": self.model, "stream": True}, stream=True) as r:
            for line in r:
                ev = json.loads(line)
                if ev.get("total"):
                    print(f"\r  {ev.get('status', '')} {ev.get('completed', 0) * 100 // ev['total']:3d}%",
                          end="", file=sys.stderr)
                if ev.get("error"):
                    die(f"Ollama pull failed: {ev['error']}")
        print(file=sys.stderr)

    # -- common API
    def count_tokens(self, text: str) -> int:
        if self.backend == "llamacpp":
            return len(self.llm.tokenize(text.encode(), add_bos=False))
        return len(text) // 3  # rough estimate; Ollama has no tokenize endpoint

    def json_chat(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        # "/no_think" turns off Qwen3's reasoning preamble; other models ignore it.
        user = user + "\n/no_think"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "llamacpp":
            out = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object", "schema": schema},
                temperature=0.1,
                max_tokens=max_tokens,
            )
            text = out["choices"][0]["message"]["content"]
            finish = out["choices"][0].get("finish_reason")
        else:
            out = self._ollama("/api/chat", {
                "model": self.model, "messages": messages, "format": schema, "stream": False,
                "options": {"num_ctx": self.n_ctx, "temperature": 0.1, "num_predict": max_tokens},
            })
            text = out["message"]["content"]
            finish = out.get("done_reason")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            hint = " (output hit the token limit; try a larger --ctx)" if finish == "length" else ""
            die(f"The model did not return valid JSON{hint}.")


def transcript_lines(utterances: list[dict]) -> list[str]:
    return [f"#{u['id']} [{fmt_time(u['start_ms'])}] Speaker {u['speaker']}: {u['text']}" for u in utterances]


def analyse(utterances: list[dict], llm: LocalLLM, known_names: dict[str, str]) -> dict:
    names_hint = ""
    if known_names:
        names_hint = "\nKnown speaker identities (authoritative): " + "; ".join(
            f"Speaker {k} = {v}" for k, v in known_names.items()) + "\n"

    lines = transcript_lines(utterances)
    output_budget = min(8192, llm.n_ctx // 4)
    input_budget = llm.n_ctx - output_budget - llm.count_tokens(SYSTEM_PROMPT) - 200

    # Split long meetings into chunks that fit the context window.
    chunks, cur, cur_tokens = [], [], 0
    for line in lines:
        t = llm.count_tokens(line) + 1
        if cur and cur_tokens + t > input_budget:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(line)
        cur_tokens += t
    if cur:
        chunks.append(cur)

    results = []
    for i, chunk in enumerate(chunks, 1):
        part = f" (part {i} of {len(chunks)}; analyse only this part)" if len(chunks) > 1 else ""
        log(f"Analysing transcript{part} with the local LLM - this can take a few minutes on CPU...")
        t0 = time.time()
        results.append(llm.json_chat(
            SYSTEM_PROMPT,
            f"<transcript>\n" + "\n".join(chunk) + f"\n</transcript>\n{names_hint}\nProduce the meeting minutes{part}.",
            ANALYSIS_SCHEMA,
            output_budget,
        ))
        log(f"  done in {time.time() - t0:.0f}s")

    return results[0] if len(results) == 1 else merge_chunks(results, llm)


def merge_chunks(results: list[dict], llm: LocalLLM) -> dict:
    merged = {k: [] for k in ("discussion_points", "decisions", "action_items", "open_questions")}
    speakers: dict[str, dict] = {}
    for r in results:
        for k in merged:
            merged[k].extend(r.get(k, []))
        for s in r.get("speakers", []):
            prev = speakers.get(s.get("label", ""))
            if prev is None or (not prev.get("inferred_name") and s.get("inferred_name")):
                speakers[s.get("label", "")] = s
    merged["speakers"] = list(speakers.values())

    log("Writing the overall summary...")
    parts = "\n\n".join(f"Part {i}: {r.get('summary', '')}" for i, r in enumerate(results, 1))
    top = llm.json_chat(
        "You write a title and a short overall summary for a meeting, given summaries of its consecutive parts.",
        parts, SUMMARY_SCHEMA, 1024,
    )
    merged.update(title=top.get("title", "Meeting"), summary=top.get("summary", ""))
    return merged


# --------------------------------------------------------------------------
# 5. Resolve citations to timestamps + render


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

    def who(value: str) -> str:
        """Speaker label -> display name; a plain name is passed through."""
        value = str(value or "").strip()
        label = value.upper().removeprefix("SPEAKER ").strip()
        if label in names:
            return names[label]
        return value

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

    a["duration"] = fmt_time(utterances[-1]["end_ms"])
    return a


def render_markdown(r: dict) -> str:
    def times(refs):
        return ", ".join(f"`{x['time']}`" for x in refs) or "_no reference_"

    def cell(s):
        return str(s or "—").replace("|", "\\|").replace("\n", " ")

    people = ", ".join(s["name"] + (f" ({s['role']})" if s.get("role") else "") for s in r["speakers"])
    md = [f"# {r.get('title', 'Meeting')}", "",
          f"**Duration:** {r['duration']}  ", f"**Participants:** {people}", "",
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
    p.add_argument("--speakers", type=int, help="number of speakers, if known (improves diarization)")
    p.add_argument("--names", help='map speaker labels to names, e.g. "A=Nizar,B=Sara"')
    p.add_argument("--language", help="spoken language code, e.g. en, ar, fr (default: auto-detect)")
    p.add_argument("--out", help="output path prefix (default: next to the input)")
    p.add_argument("--yes", "-y", action="store_true", help="download missing models without asking")
    p.add_argument("--no-analysis", action="store_true", help="only produce the diarized transcript")

    g = p.add_argument_group("speech models")
    g.add_argument("--whisper", default=DEFAULT_WHISPER, help=f"faster-whisper model (default {DEFAULT_WHISPER})")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--cluster-threshold", type=float, default=0.5,
                   help="speaker clustering threshold when --speakers is not given; "
                        "lower = more speakers (default 0.5)")

    g = p.add_argument_group("LLM")
    g.add_argument("--llm", default="llamacpp", choices=["llamacpp", "ollama"])
    g.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO, help="Hugging Face repo with the GGUF model")
    g.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE, help="GGUF file name in that repo, or a local path")
    g.add_argument("--gpu-layers", type=int, default=-1, help="layers to offload to GPU (-1 = all, 0 = CPU only)")
    g.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    g.add_argument("--ollama-url", default="http://localhost:11434")
    g.add_argument("--ctx", type=int, default=DEFAULT_CTX, help=f"LLM context window in tokens (default {DEFAULT_CTX})")
    args = p.parse_args()

    global ASSUME_YES
    ASSUME_YES = args.yes
    known_names = parse_names(args.names)

    if args.transcript:
        utterances = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        base = re.sub(r"(\.transcript)?\.json$", "", args.transcript)
    elif args.audio:
        if not Path(args.audio).is_file():
            die(f"Audio file not found: {args.audio}")
        log("Decoding audio...")
        audio = decode_audio(args.audio)
        # Ask about every missing model up front, so the run isn't interrupted halfway.
        ensure_whisper(args.whisper)
        ensure_diarization_models()
        if not args.no_analysis and args.llm == "llamacpp":
            ensure_gguf(args.gguf_repo, args.gguf_file)

        words = transcribe(audio, args.whisper, args.language, args.device)
        turns = diarize(audio, args.speakers, args.cluster_threshold)
        assign_speakers(words, turns)
        utterances = build_utterances(words)
        base = args.audio

        Path(f"{base}.transcript.json").write_text(json.dumps(utterances, indent=2, ensure_ascii=False), encoding="utf-8")
        Path(f"{base}.transcript.txt").write_text("\n".join(transcript_lines(utterances)) + "\n", encoding="utf-8")
        n_spk = len({u["speaker"] for u in utterances})
        log(f"Transcript: {len(utterances)} utterances, {n_spk} speakers -> {base}.transcript.txt")
    else:
        p.print_help()
        sys.exit(1)

    if not utterances:
        die("The transcript is empty - nothing to analyse.")
    if args.no_analysis:
        return

    llm = LocalLLM(args)
    analysis = analyse(utterances, llm, known_names)
    report = attach_timestamps(analysis, utterances, known_names)

    out = args.out or base
    Path(f"{out}.actions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(f"{out}.actions.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"Wrote {out}.actions.md and {out}.actions.json")


if __name__ == "__main__":
    main()
