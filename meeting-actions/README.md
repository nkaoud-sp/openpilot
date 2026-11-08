# meeting-actions

Turn a meeting recording (mp3, m4a, wav, mp4, …) into **speaker-attributed, timestamped
minutes and action points**. It is built for long (1–2 hour), **bilingual Arabic + English**
meetings. Everything runs on your machine using open-source models, and no audio or text leaves it.

```
audio ─► speaker turns (sherpa-onnx) ─► speech pieces ─► language per piece (Whisper base, ar/en)
      ─► transcription with the language fixed per piece (Whisper large-v3-turbo, batched)
      ─► utterances: #12 [00:14:03] Speaker B: …
      ─► local LLM (llama.cpp / Ollama), in parts for long meetings, duplicates merged
      ─► each point cites utterance numbers → script adds real timestamps + quotes
      ─► meeting.actions.md / meeting.actions.json
```

## Why the language step matters

Plain Whisper detects the language once, from the first 30 seconds, and uses it for the
whole recording. In an Arabic + English meeting that means one language gets **dropped or
translated** into the other. Whisper's built-in multilingual mode doesn't solve this either
(tested). This script instead:

1. cuts the audio into speech pieces at pauses and at speaker changes,
2. detects each piece's language with a small Whisper model, limited to `--languages ar,en`,
3. transcribes each piece with its language fixed.

Arabic is written in Arabic script and English in Latin script, with timestamps intact.
A single sentence that mixes both languages is transcribed in its dominant language.

## Models (all ungated, downloaded only after you say yes)

| Step | Model | License | Size |
|---|---|---|---|
| Speech-to-text | Whisper `large-v3-turbo` (faster-whisper) | MIT | 1.5 GB |
| Language detection | Whisper `base` | MIT | 0.14 GB |
| Speaker segmentation | pyannote segmentation-3.0 (ONNX) | MIT | 7 MB |
| Speaker voice prints | 3D-Speaker ERes2Net (ONNX) | Apache-2.0 | 40 MB |
| Minutes / action points | Qwen3-8B Q4_K_M GGUF (llama.cpp); reads Arabic and English | Apache-2.0 | 4.7 GB |

The first time a model is needed, the script asks before downloading anything:

```
The model 'Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf' (4.7 GB) is not on this machine. Download it now? [y/N]
```

## Setup

```bash
cd meeting-actions
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

- **NVIDIA GPU (strongly recommended for 1–2 h meetings):** faster-whisper uses CUDA automatically
  once cuBLAS/cuDNN are installed. Build llama.cpp with CUDA support using
  `CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python`.
- **Apple Silicon:** llama-cpp-python uses Metal. Whisper and speaker detection run on the CPU.
- **Already use Ollama?** Pass `--llm ollama` instead of installing llama-cpp-python.

## Usage

```bash
# typical bilingual meeting
python meeting_actions.py meeting.m4a --languages ar,en --speakers 5

# transcript first, check who is A/B/C, then write the minutes with names
python meeting_actions.py meeting.mp3 --languages ar,en --speakers 5 --no-analysis
python meeting_actions.py --transcript meeting.mp3.transcript.json --names "A=Nizar,B=Sara,C=Omar"

# minutes written in Arabic instead of English
python meeting_actions.py --transcript meeting.mp3.transcript.json --minutes-language Arabic
```

| Option | Meaning |
|---|---|
| `--languages ar,en` | Languages spoken. Without it, they are detected automatically. Setting it avoids confusion with e.g. Persian or Urdu. |
| `--speakers N` | Number of people. **The biggest accuracy boost for speaker separation.** |
| `--names "A=Nizar,B=Sara"` | Speaker labels → names. Unmapped speakers get names the LLM finds in the conversation. |
| `--minutes-language` | Language of the written minutes (default English). Quotes stay as spoken. |
| `--whisper large-v3` | Somewhat better transcription, about 2× slower than `large-v3-turbo`. |
| `--chunk-tokens 12000` | How much transcript the LLM reads per pass (~30–40 min of talk). |
| `--batch-size 8` | Whisper batch size. Lower it if you run out of (GPU) memory. |
| `--fresh` | Ignore checkpoints and redo everything. |

## Long meetings

- **Checkpoints:** speaker detection and transcription are saved next to the recording
  (`*.speakers.ckpt.json`, `*.words.ckpt.json`). If a run is interrupted, or you only change
  LLM options, re-running reuses them. Use `--fresh` to start over.
- **Parts:** a 1–2 hour transcript is split into ~30–40 minute parts at the longest pause near
  each boundary. Each part is analysed, then a merge pass combines duplicate action items,
  decisions and open questions across parts, keeping every citation. A final pass writes the
  overall title and summary.
- **Progress:** each step shows its progress and an estimated time left.

### How long does it take?

Measured on a 4-core CPU with no GPU:

| Step | Speed | 1 h meeting | 2 h meeting |
|---|---|---|---|
| Speaker detection | ~5× real time | ~12 min | ~25 min |
| Language detection + transcription (`large-v3-turbo`) | see `examples/` timing | | |
| LLM (Qwen3-8B) | ~4 min per part + merging | ~15 min | ~25 min |

An NVIDIA GPU makes transcription and the LLM 10× faster or more. Speaker detection stays on the CPU.

## Output

- `*.transcript.txt` / `.json`: the diarized transcript (`#12 [00:03:41] Speaker B: …`); the JSON
  also records each utterance's language.
- `*.actions.md`: summary, participants with speaking time, an action-item table (owner,
  due date, priority, timestamps), decisions, numbered discussion points with quotes, and open questions.
- `*.actions.json`: the same data. Every item has `references` with `time`, `start_ms`, `end_ms`,
  `speaker` and `quote`.

## Limitations

- Speaker separation is the weakest step. Very short turns ("yes", "تمام"), people talking over
  each other, and similar voices get mixed up. `--speakers` helps a lot.
- Sentences that switch language mid-way are transcribed in one language. English terms inside
  Arabic speech may come out in Arabic script.
- The 8B model sometimes misses an action item or guesses a priority. For better minutes, use a
  bigger GGUF if you have the memory: `--gguf-repo Qwen/Qwen3-14B-GGUF --gguf-file Qwen3-14B-Q4_K_M.gguf`.
