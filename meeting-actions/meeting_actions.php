<?php

declare(strict_types=1);

/**
 * meeting_actions.php
 *
 * Turn a recorded meeting into speaker-attributed, timestamped action points.
 *
 *   1. Transcribe + diarize the audio with AssemblyAI (who spoke when).
 *   2. Send the numbered, timestamped transcript to Claude, which returns
 *      structured JSON: summary, speakers, discussion points, decisions,
 *      action items, open questions. Every point cites utterance numbers.
 *   3. Map those utterance numbers back to real timestamps (done in PHP, so
 *      the timestamps come from the transcript, never invented by the model)
 *      and write a Markdown report + a JSON file.
 *
 * Usage:
 *   php meeting_actions.php meeting.mp3
 *   php meeting_actions.php meeting.m4a --speakers=3 --names="A=Nizar,B=Sara"
 *   php meeting_actions.php --transcript=meeting.transcript.json   (skip transcription)
 *
 * Env:
 *   ASSEMBLYAI_API_KEY   for transcription
 *   ANTHROPIC_API_KEY    for the analysis
 */

require __DIR__ . '/vendor/autoload.php';

use Anthropic\Client;
use Anthropic\Core\Exceptions\APIConnectionException;
use Anthropic\Core\Exceptions\APIStatusException;
use Anthropic\Core\Exceptions\AuthenticationException;
use Anthropic\Core\Exceptions\BadRequestException;
use Anthropic\Core\Exceptions\RateLimitException;

const CLAUDE_MODEL = 'claude-opus-5-5';
const ASSEMBLY_BASE = 'https://api.assemblyai.com/v2';

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    main($argv);
}

// ---------------------------------------------------------------------------

function main(array $argv): void
{
    $opts = parseArgs($argv);

    if ($opts['transcript'] !== null) {
        $utterances = loadTranscript($opts['transcript']);
        $base = preg_replace('/\.transcript\.json$|\.json$/', '', $opts['transcript']);
    } else {
        if ($opts['audio'] === null) {
            usage();
        }
        $utterances = transcribe($opts['audio'], $opts['speakers'], $opts['language']);
        $base = $opts['audio'];
        // Cache the transcript so you can re-run the analysis without paying for transcription again.
        file_put_contents("$base.transcript.json", json_encode($utterances, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
        logLine("Transcript saved to $base.transcript.json");
    }

    if (!$utterances) {
        fail('Transcript is empty - nothing to analyse.');
    }

    logLine(sprintf('Analysing %d utterances with Claude...', count($utterances)));
    $analysis = analyse($utterances, $opts['names']);

    $report = attachTimestamps($analysis, $utterances, $opts['names']);

    $out = $opts['out'] ?? $base;
    file_put_contents("$out.actions.json", json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    file_put_contents("$out.actions.md", renderMarkdown($report));
    logLine("Wrote $out.actions.md and $out.actions.json");
}

// ---------------------------------------------------------------------------
// CLI

function parseArgs(array $argv): array
{
    $opts = ['audio' => null, 'transcript' => null, 'speakers' => null,
             'names' => [], 'language' => null, 'out' => null];

    foreach (array_slice($argv, 1) as $arg) {
        if (preg_match('/^--(\w+)=(.*)$/s', $arg, $m)) {
            [$k, $v] = [$m[1], $m[2]];
            match ($k) {
                'transcript' => $opts['transcript'] = $v,
                'speakers'   => $opts['speakers'] = (int) $v,
                'language'   => $opts['language'] = $v,
                'out'        => $opts['out'] = $v,
                'names'      => $opts['names'] = parseNames($v),
                default      => fail("Unknown option --$k"),
            };
        } elseif ($arg === '-h' || $arg === '--help') {
            usage();
        } else {
            $opts['audio'] = $arg;
        }
    }
    return $opts;
}

function parseNames(string $v): array
{
    $names = [];
    foreach (array_filter(array_map('trim', explode(',', $v))) as $pair) {
        [$label, $name] = array_map('trim', explode('=', $pair, 2)) + [1 => ''];
        if ($label !== '' && $name !== '') {
            $names[strtoupper($label)] = $name;
        }
    }
    return $names;
}

function usage(): never
{
    fwrite(STDERR, <<<TXT
    Usage:
      php meeting_actions.php <audio-file> [options]
      php meeting_actions.php --transcript=<file.transcript.json> [options]

    Options:
      --speakers=N          Expected number of speakers (improves diarization)
      --names="A=Nizar,B=Sara"  Map diarized speaker labels to real names
      --language=en         Language code (default: auto-detect)
      --out=path/prefix     Output prefix (default: next to the input file)

    TXT);
    exit(1);
}

// ---------------------------------------------------------------------------
// Step 1: transcription + speaker diarization (AssemblyAI)

/**
 * @return list<array{id:int, speaker:string, start_ms:int, end_ms:int, text:string}>
 */
function transcribe(string $path, ?int $speakers, ?string $language): array
{
    if (!is_file($path)) {
        fail("Audio file not found: $path");
    }
    $key = getenv('ASSEMBLYAI_API_KEY') ?: fail('Set ASSEMBLYAI_API_KEY');

    logLine('Uploading audio...');
    $upload = http('POST', ASSEMBLY_BASE . '/upload', $key, fopen($path, 'rb'), filesize($path));

    $request = ['audio_url' => $upload['upload_url'], 'speaker_labels' => true];
    if ($speakers) {
        $request['speakers_expected'] = $speakers;
    }
    if ($language) {
        $request['language_code'] = $language;
    } else {
        $request['language_detection'] = true;
    }

    $job = http('POST', ASSEMBLY_BASE . '/transcript', $key, json_encode($request));
    logLine("Transcribing (job {$job['id']})...");

    $delay = 3;
    while (true) {
        sleep($delay);
        $job = http('GET', ASSEMBLY_BASE . "/transcript/{$job['id']}", $key);
        if ($job['status'] === 'completed') {
            break;
        }
        if ($job['status'] === 'error') {
            fail('Transcription failed: ' . ($job['error'] ?? 'unknown error'));
        }
        $delay = min($delay + 2, 15);
    }

    $utterances = [];
    foreach ($job['utterances'] ?? [] as $i => $u) {
        $utterances[] = [
            'id'       => $i + 1,
            'speaker'  => (string) $u['speaker'],
            'start_ms' => (int) $u['start'],
            'end_ms'   => (int) $u['end'],
            'text'     => trim($u['text']),
        ];
    }
    logLine(sprintf('Got %d utterances from %d speakers.',
        count($utterances), count(array_unique(array_column($utterances, 'speaker')))));
    return $utterances;
}

/**
 * Minimal JSON-over-HTTPS helper for AssemblyAI.
 *
 * @param resource|string|null $body  a stream (file upload) or a JSON string
 */
function http(string $method, string $url, string $key, $body = null, ?int $size = null): array
{
    $ch = curl_init($url);
    $headers = ["Authorization: $key"];
    curl_setopt_array($ch, [
        CURLOPT_CUSTOMREQUEST  => $method,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 600,
    ]);
    if (is_resource($body)) {
        $headers[] = 'Content-Type: application/octet-stream';
        curl_setopt_array($ch, [CURLOPT_UPLOAD => true, CURLOPT_INFILE => $body,
                                CURLOPT_INFILESIZE => $size, CURLOPT_CUSTOMREQUEST => 'POST']);
    } elseif (is_string($body)) {
        $headers[] = 'Content-Type: application/json';
        curl_setopt($ch, CURLOPT_POSTFIELDS, $body);
    }
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

    $raw = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    $err = curl_error($ch);
    curl_close($ch);

    if ($raw === false) {
        fail("HTTP $method $url failed: $err");
    }
    $data = json_decode($raw, true);
    if ($status >= 400 || !is_array($data)) {
        fail("HTTP $method $url returned $status: $raw");
    }
    return $data;
}

function loadTranscript(string $path): array
{
    $data = json_decode((string) @file_get_contents($path), true);
    if (!is_array($data)) {
        fail("Could not read transcript JSON: $path");
    }
    return $data;
}

// ---------------------------------------------------------------------------
// Step 2: analysis with Claude (structured output)

function analyse(array $utterances, array $knownNames): array
{
    $client = new Client(); // reads ANTHROPIC_API_KEY

    $lines = [];
    foreach ($utterances as $u) {
        $lines[] = sprintf('#%d [%s] Speaker %s: %s',
            $u['id'], fmtTime($u['start_ms']), $u['speaker'], $u['text']);
    }
    $transcript = implode("\n", $lines);

    $namesHint = '';
    if ($knownNames) {
        $pairs = [];
        foreach ($knownNames as $label => $name) {
            $pairs[] = "Speaker $label = $name";
        }
        $namesHint = "\nKnown speaker identities (authoritative): " . implode('; ', $pairs) . "\n";
    }

    $system = <<<SYS
    You analyse meeting transcripts and turn them into clear, actionable minutes.

    The transcript was produced by automatic speech recognition with speaker
    diarization. Each line is: #<utterance number> [<timestamp>] Speaker <label>: <text>.
    Expect recognition errors and occasional mis-attributed speakers; use context
    to interpret them sensibly, and do not invent content that was not said.

    What to produce:
    - speakers: for each diarized label, infer the person's name if the
      conversation reveals it (people addressing each other, introductions),
      and their apparent role. Leave inferred_name empty if it is not evident.
    - discussion_points: break the meeting into the distinct topics in the order
      they were discussed, each with a concise summary of what was said and by whom.
    - decisions: things the group actually agreed or settled.
    - action_items: concrete follow-ups someone committed to or was assigned.
      Owner is the speaker label of the person responsible (empty if unassigned).
      Capture deadlines exactly as stated ("by Friday"), or empty if none.
    - open_questions: issues raised but left unresolved.

    For every item, cite the utterance numbers (the #N values) that support it,
    most relevant first. Cite only numbers that appear in the transcript.
    SYS;

    $item = fn (array $props) => [
        'type' => 'object',
        'properties' => $props + ['utterance_ids' => ['type' => 'array', 'items' => ['type' => 'integer']]],
        'required' => array_merge(array_keys($props), ['utterance_ids']),
        'additionalProperties' => false,
    ];
    $str = ['type' => 'string'];

    $schema = [
        'type' => 'object',
        'properties' => [
            'title'   => $str,
            'summary' => $str,
            'speakers' => ['type' => 'array', 'items' => [
                'type' => 'object',
                'properties' => ['label' => $str, 'inferred_name' => $str, 'role' => $str],
                'required' => ['label', 'inferred_name', 'role'],
                'additionalProperties' => false,
            ]],
            'discussion_points' => ['type' => 'array', 'items' => $item(['topic' => $str, 'summary' => $str])],
            'decisions'         => ['type' => 'array', 'items' => $item(['decision' => $str])],
            'action_items'      => ['type' => 'array', 'items' => $item([
                'task'      => $str,
                'owner'     => $str,
                'due'       => $str,
                'priority'  => ['type' => 'string', 'enum' => ['high', 'medium', 'low']],
            ])],
            'open_questions'    => ['type' => 'array', 'items' => $item(['question' => $str])],
        ],
        'required' => ['title', 'summary', 'speakers', 'discussion_points', 'decisions', 'action_items', 'open_questions'],
        'additionalProperties' => false,
    ];

    try {
        $message = $client->beta->messages->create(
            model: CLAUDE_MODEL,
            maxTokens: 16000,
            system: $system,
            messages: [[
                'role' => 'user',
                'content' => "<transcript>\n$transcript\n</transcript>\n$namesHint\nProduce the meeting minutes.",
            ]],
            outputConfig: [
                'effort' => 'high',
                'format' => ['type' => 'json_schema', 'schema' => $schema],
            ],
            // If a safety classifier declines, retry server-side on a fallback model.
            betas: ['server-side-fallback-2026-07-01'],
            fallbacks: 'default',
        );
    } catch (AuthenticationException) {
        fail('Anthropic API key is missing or invalid (set ANTHROPIC_API_KEY).');
    } catch (BadRequestException $e) {
        fail('Claude rejected the request: ' . $e->getMessage());
    } catch (RateLimitException) {
        fail('Rate limited by the Anthropic API - try again in a minute.');
    } catch (APIStatusException $e) {
        fail('Anthropic API error: ' . $e->getMessage());
    } catch (APIConnectionException $e) {
        fail('Could not reach the Anthropic API: ' . $e->getMessage());
    }

    if ($message->stopReason === 'refusal') {
        fail('Claude declined to process this transcript.');
    }
    if ($message->stopReason === 'max_tokens') {
        fail('Output was cut off (meeting too long for one pass). Split the transcript and re-run.');
    }

    foreach ($message->content as $block) {
        if ($block->type === 'text') {
            $data = json_decode($block->text, true);
            if (is_array($data)) {
                return $data;
            }
        }
    }
    fail('Claude did not return valid JSON.');
}

// ---------------------------------------------------------------------------
// Step 3: resolve citations to timestamps + render

function attachTimestamps(array $a, array $utterances, array $knownNames): array
{
    $byId = [];
    foreach ($utterances as $u) {
        $byId[$u['id']] = $u;
    }

    // Speaker label -> display name: CLI names win, then names Claude inferred.
    $names = [];
    foreach ($a['speakers'] as $s) {
        $label = strtoupper($s['label']);
        $names[$label] = $knownNames[$label] ?? ($s['inferred_name'] !== '' ? $s['inferred_name'] : "Speaker $label");
    }
    foreach ($knownNames as $label => $name) {
        $names[$label] ??= $name;
    }
    $who = fn (string $label) => $label === '' ? '' : ($names[strtoupper($label)] ?? "Speaker $label");

    $refs = function (array $ids) use ($byId, $who): array {
        $out = [];
        foreach (array_unique($ids) as $id) {
            if (!isset($byId[$id])) {
                continue; // drop any citation that doesn't exist
            }
            $u = $byId[$id];
            $out[] = [
                'utterance' => $id,
                'time'      => fmtTime($u['start_ms']),
                'start_ms'  => $u['start_ms'],
                'end_ms'    => $u['end_ms'],
                'speaker'   => $who($u['speaker']),
                'quote'     => $u['text'],
            ];
        }
        usort($out, fn ($x, $y) => $x['start_ms'] <=> $y['start_ms']);
        return $out;
    };

    foreach (['discussion_points', 'decisions', 'action_items', 'open_questions'] as $section) {
        foreach ($a[$section] as &$entry) {
            $entry['references'] = $refs($entry['utterance_ids']);
            unset($entry['utterance_ids']);
        }
        unset($entry);
    }
    foreach ($a['action_items'] as &$ai) {
        $ai['owner'] = $who($ai['owner']);
    }
    unset($ai);

    foreach ($a['speakers'] as &$s) {
        $s['name'] = $names[strtoupper($s['label'])];
    }
    unset($s);

    $last = end($utterances);
    $a['duration'] = fmtTime($last['end_ms']);
    return $a;
}

function renderMarkdown(array $r): string
{
    $times = fn (array $refs) => $refs
        ? implode(', ', array_map(fn ($x) => "`{$x['time']}`", $refs))
        : '_no reference_';

    $md = "# {$r['title']}\n\n";
    $md .= "**Duration:** {$r['duration']}  \n";
    $md .= '**Participants:** ' . implode(', ', array_map(
        fn ($s) => $s['name'] . ($s['role'] !== '' ? " ({$s['role']})" : ''), $r['speakers'])) . "\n\n";
    $md .= "## Summary\n\n{$r['summary']}\n\n";

    $md .= "## Action items\n\n";
    if (!$r['action_items']) {
        $md .= "_None recorded._\n\n";
    } else {
        $md .= "| # | Action | Owner | Due | Priority | Timestamps |\n|---|---|---|---|---|---|\n";
        foreach ($r['action_items'] as $i => $ai) {
            $md .= sprintf("| %d | %s | %s | %s | %s | %s |\n", $i + 1, cell($ai['task']),
                cell($ai['owner'] ?: '—'), cell($ai['due'] ?: '—'), $ai['priority'], $times($ai['references']));
        }
        $md .= "\n";
    }

    $md .= "## Decisions\n\n";
    foreach ($r['decisions'] as $d) {
        $md .= "- {$d['decision']} — {$times($d['references'])}\n";
    }
    $md .= $r['decisions'] ? "\n" : "_None recorded._\n\n";

    $md .= "## Discussion points\n\n";
    foreach ($r['discussion_points'] as $i => $p) {
        $md .= sprintf("### %d. %s  (%s)\n\n%s\n\n", $i + 1, $p['topic'], $times($p['references']), $p['summary']);
        foreach (array_slice($p['references'], 0, 3) as $ref) {
            $md .= "> `{$ref['time']}` **{$ref['speaker']}:** {$ref['quote']}\n>\n";
        }
        $md = rtrim($md, ">\n") . "\n\n";
    }

    if ($r['open_questions']) {
        $md .= "## Open questions\n\n";
        foreach ($r['open_questions'] as $q) {
            $md .= "- {$q['question']} — {$times($q['references'])}\n";
        }
        $md .= "\n";
    }
    return $md;
}

// ---------------------------------------------------------------------------
// helpers

function fmtTime(int $ms): string
{
    $s = intdiv($ms, 1000);
    return sprintf('%02d:%02d:%02d', intdiv($s, 3600), intdiv($s % 3600, 60), $s % 60);
}

function cell(string $s): string
{
    return str_replace(['|', "\n"], ['\\|', ' '], $s);
}

function logLine(string $msg): void
{
    fwrite(STDERR, "[meeting-actions] $msg\n");
}

function fail(string $msg): never
{
    fwrite(STDERR, "Error: $msg\n");
    exit(1);
}
