# meeting-actions

Turn a meeting recording into **speaker-attributed, timestamped minutes and action points**.
Everything runs on your machine using open-source models. No audio or text leaves it.

```
audio ─► faster-whisper ─────────► words + timestamps ─┐
      └► sherpa-onnx diarization ─► who spoke when ─────┴─► utterances (#1 [00:01:23] Speaker A: …)
                                                             │
                     local LLM (llama.cpp / Ollama) ◄────────┘
                       → topics, decisions, action items, open questions,
                         each citing utterance numbers
                                                             │
          script maps citations → real timestamps + quotes ◄─┘
          → meeting.actions.md / meeting.actions.json
```

The LLM never writes timestamps. It cites utterance numbers, and the script looks
up their real start/end times. Citations to utterances that don't exist are dropped.

## Models (all ungated, downloaded only after you say yes)

| Step | Model | License | Size |
|---|---|---|---|
| Speech-to-text | Whisper `large-v3-turbo` via faster-whisper (`--whisper small` is faster on CPU) | MIT | 1.6 GB (small: 0.46 GB) |
| Speaker segmentation | pyannote segmentation-3.0 (ONNX export by sherpa-onnx) | MIT | 7 MB |
| Speaker embeddings | 3D-Speaker ERes2Net (ONNX) | Apache-2.0 | 40 MB |
| Minutes / action points | Qwen3-8B Q4_K_M GGUF via llama.cpp | Apache-2.0 | 5 GB |

The first time a model is needed, the script asks:

```
The model 'Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf' (4.7 GB) is not on this machine. Download it now? [y/N]
```

Pass `--yes` to download without asking, e.g. in automation. Whisper and GGUF models go to the
Hugging Face cache. The diarization models go to `~/.cache/meeting-actions` (override with `MEETING_ACTIONS_CACHE`).

## Setup

```bash
cd meeting-actions
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

- **GPU (NVIDIA):** faster-whisper uses CUDA automatically if cuBLAS/cuDNN are installed. Build llama.cpp with CUDA using
  `CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python`.
- **Apple Silicon:** llama-cpp-python uses Metal by default. Whisper runs on the CPU.
- **Already use Ollama?** Skip `llama-cpp-python` and pass `--llm ollama`. The script offers to `ollama pull` the model.

## Usage

```bash
python meeting_actions.py meeting.mp3
python meeting_actions.py meeting.m4a --speakers 3 --names "A=Nizar,B=Sara"
python meeting_actions.py meeting.mp3 --no-analysis            # diarized transcript only
python meeting_actions.py --transcript meeting.mp3.transcript.json   # re-run only the LLM step
python meeting_actions.py meeting.mp3 --llm ollama --ollama-model qwen3:8b
```

| Option | Meaning |
|---|---|
| `--speakers N` | Number of people, if known. This is the single biggest accuracy boost for speaker separation. |
| `--names "A=Nizar,B=Sara"` | Map speaker labels to names. Unmapped labels use names the LLM infers from the conversation, or stay "Speaker C". |
| `--language ar` | Force the spoken language (default: auto-detect). |
| `--cluster-threshold 0.5` | Without `--speakers`: lower finds more speakers, higher finds fewer. |
| `--whisper NAME` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, or a local model folder. |
| `--gguf-repo / --gguf-file` | Any GGUF chat model from Hugging Face, or a local `.gguf` path. |
| `--ctx 32768` | LLM context size. Longer meetings are split into parts automatically. |
| `--gpu-layers 0` | Force the LLM onto the CPU. |

A practical workflow is to run once with `--no-analysis`, check `*.transcript.txt` to see which
letter is who, then re-run the analysis with `--transcript … --names …`.

## Output

- `meeting.mp3.transcript.txt` / `.json`: the diarized transcript (`#12 [00:03:41] Speaker B: …`).
- `meeting.mp3.actions.md`: summary, a table of action items (owner, due date, priority,
  timestamps), decisions, numbered discussion points with supporting quotes, and open questions.
- `meeting.mp3.actions.json`: the same data. Every item has `references` with `time`, `start_ms`,
  `end_ms`, `speaker` and `quote`.

## Limitations

- Speaker separation is the weakest step, especially for very short turns ("yes", "agreed"),
  people talking over each other, and similar-sounding voices. Giving `--speakers` helps a lot.
- On CPU, expect roughly 0.3–1× real time for `large-v3-turbo` transcription plus a few minutes
  for the LLM. A GPU makes both much faster.
- Very long meetings are analysed in parts and then merged. Duplicate action items across parts
  are not removed automatically.

## Example

`examples/launch-sync.mp3` is a 48-second, three-person meeting generated with Piper TTS voices.
Running `python meeting_actions.py examples/launch-sync.mp3 --whisper small --speakers 3` on a
4-core CPU (no GPU) takes about 5 minutes. The resulting transcript and minutes are next to it in `examples/`.
