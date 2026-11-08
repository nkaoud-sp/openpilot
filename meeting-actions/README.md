# meeting-actions

PHP CLI that turns a recorded meeting into **speaker-attributed, timestamped minutes and action points**.

```
audio ──► AssemblyAI (speech-to-text + speaker diarization)
      ──► Claude (structured JSON: topics, decisions, action items, open questions)
      ──► PHP maps every cited utterance back to its real timestamp
      ──► meeting.actions.md + meeting.actions.json
```

Claude never writes timestamps itself. It cites utterance numbers (`#12`), and the
script looks up each number's start/end time in the transcript. If Claude cites a
number that doesn't exist, the script drops it.

## Setup

```bash
cd meeting-actions
composer install                       # PHP 8.1+, ext-curl
export ASSEMBLYAI_API_KEY=...          # https://www.assemblyai.com
export ANTHROPIC_API_KEY=...           # https://platform.claude.com
```

## Usage

```bash
php meeting_actions.php meeting.mp3
php meeting_actions.php meeting.m4a --speakers=3 --names="A=Nizar,B=Sara"
php meeting_actions.php --transcript=meeting.mp3.transcript.json   # re-analyse, no re-transcription
```

| Option | Meaning |
|---|---|
| `--speakers=N` | Expected speaker count. Improves diarization accuracy. |
| `--names="A=Nizar,B=Sara"` | Map diarized labels to real names. Labels you leave out get names Claude infers from the conversation, or stay "Speaker C". |
| `--language=en` | Force a language (default: auto-detect). |
| `--out=prefix` | Output path prefix. |

## Output

- `*.transcript.json`: the diarized transcript, cached so you can re-run the analysis without paying for transcription again.
- `*.actions.md`: a readable report with a summary, an action-item table (owner, due date, priority, timestamps), decisions, numbered discussion points with supporting quotes, and open questions.
- `*.actions.json`: the same data for integrations. Every item has a `references` list with `time`, `start_ms`, `end_ms`, `speaker` and `quote`.

`examples/sample.transcript.json` is a small transcript in the expected format. Use it to try the analysis step without an audio file.
