# DocDrift

DocDrift is a small Python tool that detects documentation that became outdated when code changed.

This repository implements a **deterministic detector for renamed CLI flags** plus an **experimental, optional** local-model helper for semantic drift.

The deterministic detector reads a unified Git diff, infers unambiguous flag renames, and reports Markdown lines that still mention the old flag. It does not call a model.

The experimental AI helper talks only to a **local Ollama** server. It does not call cloud AI APIs. It does not execute analyzed repositories or lint documentation in general.

## Problem

Pull requests often rename command-line options in source code while README files and other docs keep the old names. Readers then follow instructions that no longer work.

Example:

- Code changes `--timeout` to `--deadline`.
- `README.md` still says `Use `--timeout` to stop slow jobs.`
- DocDrift reports that outdated sentence and links it to the code change.

## How detection works

1. Parse a unified Git diff with the Python standard library.
2. Look for source-style option declarations such as `argparse.add_argument(...)`.
3. Pair an old flag with a new flag only when a single change block gives enough evidence of a rename. Ambiguous groups of additions and deletions are skipped.
4. Recursively search `*.md` files under the supplied repository path.
5. Report boundary-aware matches of the **old** flag. Mentions of the new flag are not treated as errors.

## Technical approach

The MVP is intentionally narrow:

- Standard library only (`pathlib`, `re`, `dataclasses`, `urllib` for the optional local AI helper).
- No cloud model APIs, database, or GitHub App. The experimental AI helper uses local Ollama only.
- Findings are structured dictionaries so callers can inspect them programmatically.
- Accuracy is preferred over recall. If a rename cannot be proven, DocDrift returns nothing for that candidate.

## Repository structure

```text
docdrift/
├── .github/
│   └── workflows/
│       └── docdrift.yml         ← GitHub Actions PR workflow
├── HACKATHON.md
├── LICENSE
├── README.md
├── docdrift/
│   ├── __init__.py
│   ├── detector.py
│   ├── ai_detector.py
│   ├── orchestrator.py
│   └── github_reporter.py      ← GitHub Actions integration module
├── tests/
│   ├── test_detector.py
│   ├── test_ai_detector.py
│   ├── test_orchestrator.py
│   └── test_github_reporter.py ← GitHub integration tests
├── examples/
│   └── sample_repo/
│       ├── cli.py
│       └── README.md
└── requirements.txt
```

## Requirements

- Python 3.9 or newer
- No third-party packages (`requirements.txt` is intentionally empty of dependencies)

## Installation

This milestone has no packaging metadata and no third-party dependencies. From the repository root, import the package directly (or set `PYTHONPATH` to the repository root):

```bash
cd /path/to/Automated-Documentation-Drift-Detection-System
PYTHONPATH=. python3 -c "from docdrift.detector import detect_drift; print(detect_drift)"
```

## Usage

```python
from pathlib import Path
from docdrift.detector import detect_drift

diff = """
diff --git a/cli.py b/cli.py
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""

findings = detect_drift(diff, Path("examples/sample_repo"))
for finding in findings:
    print(finding)
```

`detect_drift` returns a list of dictionaries. It does not print results by itself.

### Example finding

Against `examples/sample_repo/`, the sample rename above produces one finding similar to:

```python
{
    "type": "renamed_cli_flag",
    "documentation_file": "README.md",
    "line_number": 3,
    "outdated_text": "Use `--timeout` to stop slow jobs.",
    "old_flag": "--timeout",
    "new_flag": "--deadline",
    "code_change": {
        "removed": 'parser.add_argument("--timeout", help="Maximum execution time")',
        "added": 'parser.add_argument("--deadline", help="Maximum execution time")',
    },
}
```

The sample README also contains a correct `--deadline` reference. That line is not reported.

The sample documentation is **left outdated on purpose**. Do not “fix” `examples/sample_repo/README.md`; it is the demonstration case.

## Unified Drift Analysis (`analyze_changes`)

DocDrift provides a top-level orchestration function, `analyze_changes`, that bridges the deterministic detector and the optional local AI semantic analyzer:

```python
from pathlib import Path
from docdrift import analyze_changes

diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
-parser.add_argument("--timeout", default=30)
+parser.add_argument("--deadline", default=60)
"""

# Offline mode: deterministic detection only (no Ollama required)
result = analyze_changes(diff, Path("examples/sample_repo"), use_ai=False)

# Hybrid mode: deterministic + local AI semantic analysis against relevant docs
result_ai = analyze_changes(
    diff,
    Path("examples/sample_repo"),
    use_ai=True,
    model="qwen3:4b",    # optional, defaults to DOCDRIFT_AI_MODEL or qwen3:4b
    timeout=60.0,        # optional timeout in seconds
)
```

### Response Structure

`analyze_changes` returns a structured dictionary:

```python
{
    "deterministic_findings": [
        # List of verified deterministic findings (e.g. renamed CLI flags)
    ],
    "ai_analysis": {
        # Raw AI analyzer output when use_ai=True and analysis succeeds, else None
        "findings": [
            {
                "type": "possible_semantic_drift",
                "description": "...",
                "evidence": "Diff: `...` | Docs: `...`",
                "suggestion": "...",
                "confidence": "high",
                "evidence_verified": True,
            }
        ],
        "limitations": [...],
        "model": "qwen3:4b",
        "status": "unverified_ai_suggestions",
    },
    "documentation_files": [
        # Relative paths of documentation files analyzed by AI
    ],
    "status": "completed",  # "completed", "no_relevant_documentation", or "ai_failed"
    "ai_error": None,       # Error message string if AI analysis failed, else None
}
```

### Deterministic Findings vs. AI Suggestions

DocDrift maintains a strict boundary between two categories of findings:
1. **Deterministic findings (`deterministic_findings`)**: Verified, proof-backed facts based on code syntax (e.g., unambiguous CLI flag renames matched in documentation). High precision, zero hallucinations.
2. **AI suggestions (`ai_analysis`)**: Probabilistic, unverified suggestions from a local LLM (`qwen3:4b`). Marked with `status: "unverified_ai_suggestions"` and `evidence_verified: bool` indicating whether quoted snippets exist verbatim in the source texts. Always require human review.

### Documentation Selection Methodology

When `use_ai=True`, DocDrift does **not** send the entire repository to Ollama. Instead, it uses a targeted relevance ranking pipeline:
1. **Signal Extraction**: Extracts changed file stems, CLI flags (`--flag`, `-f`), and modified code identifiers from the unified diff.
2. **Relevance Scoring**:
   - Files matching deterministic findings: **+100 points**
   - Files matching changed code filenames or stems: **+40 points**
   - Files containing removed/added CLI flags: **+20 points**
   - Files containing modified code identifiers: **+5 points** per identifier (up to 5)
   - Root `README.md` or `index.md` (if overlapping): **+15 points**
3. **Safety & Budget Limits**:
   - Skips hidden directories (e.g., `.git/`).
   - Ignores unreadable or binary files gracefully.
   - Caps individual doc files at **64 KB** (`MAX_FILE_BYTES`). Larger files are truncated with a marker.
   - Enforces an overall budget of **96 KB** (`MAX_TOTAL_DOC_BYTES`).
   - If no files score above 0, Ollama is not called and `status: "no_relevant_documentation"` is returned.

### Resilient Error Handling

If the local Ollama daemon is unreachable, times out, or returns invalid data:
- Deterministic findings are **never lost**; they are returned immediately.
- `status` is set to `"ai_failed"`.
- `ai_error` contains the descriptive error string.
- No uncaught network or AI exceptions bubble up to crash the caller.

## Experimental AI-assisted semantic drift

`docdrift.ai_detector.analyze_semantic_drift` is an **independent, optional helper**. The deterministic detector `detect_drift()` does not depend on it and runs entirely offline without Ollama or network access.

The AI analyzer inspects a unified Git diff against documentation to identify possible semantic inconsistencies (such as changed defaults, modified parameter behaviors, or outdated descriptions of functions). Every result is returned with `status: "unverified_ai_suggestions"` to make clear that AI findings are unverified suggestions requiring human verification. Confidence ratings are subjective model estimates rather than calibrated probabilities. An empty findings list is returned when no clear drift is identified.

### Setup

1. Install and start the local Ollama daemon:

```bash
ollama serve
```

2. Pull the default open-weight model (`qwen3:4b`):

```bash
ollama pull qwen3:4b
```

3. Confirm the local chat endpoint is reachable at `http://localhost:11434/api/chat`. All model interactions stay strictly on your local machine; repository data is never sent to cloud APIs.

### Configuration

| Setting | Default | Override |
| --- | --- | --- |
| Chat endpoint | `http://localhost:11434/api/chat` | `DOCDRIFT_OLLAMA_URL` or the `base_url=` argument |
| Model | `qwen3:4b` | `DOCDRIFT_AI_MODEL` or the `model=` argument |

Explicit function arguments take precedence over environment variables, which take precedence over defaults.

### Usage

```python
from docdrift.ai_detector import analyze_semantic_drift

result = analyze_semantic_drift(
    diff,
    documentation,
    model="qwen3:4b",                          # optional, defaults to DOCDRIFT_AI_MODEL or qwen3:4b
    base_url="http://localhost:11434/api/chat", # optional, defaults to DOCDRIFT_OLLAMA_URL or localhost
)

print(result["status"])       # always "unverified_ai_suggestions"
print(result["model"])        # e.g. "qwen3:4b"
print(result["findings"])     # list of {type, description, evidence, suggestion, confidence}
print(result["limitations"])  # list of strings
```

Empty or whitespace-only `diff` and `documentation` inputs raise `ValueError` before making any network request. Request failures, HTTP errors, and malformed model outputs raise `SemanticDriftAnalysisError` rather than silently returning empty findings.

### Running tests

Automated unit tests mock `urllib.request.urlopen` and do **not** require a running Ollama server:

```bash
python3 -m unittest discover -s tests -v
```

To run only the AI detector tests:

```bash
python3 -m unittest discover -s tests -p "test_ai_detector.py" -v
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Current limitations

- Only CLI flag **renames** are detected, not changed defaults, removed commands, or prose that drifted semantically.
- Rename evidence comes from declaration-like lines in **source files** named by the diff (currently `*.py` / `*.pyi` / `*.pyw`). Markdown and other non-source diffs are ignored.
- Change lines are used only inside a valid unified-diff hunk (`@@ -n,m +n,m @@` after file headers). Headerless `-`/`+` text cannot create findings.
- Competing declarations that share the same line template are not paired. The first match is not accepted greedily.
- Multiple flags on one declaration are paired only with structural evidence (one replacement, or one long option plus one short option). Position/`zip` pairing is not used, so two long flags replaced on the same line are skipped.
- If the same old flag is mapped to more than one new flag, that old flag is omitted.
- Markdown filenames are not treated as documentation occurrences; only file contents are scanned.
- Semantic analysis is experimental: it requires a local Ollama server, treats model output as unverified suggestions, and does not invent or confirm file/line evidence on its own.
- The GitHub Actions workflow has not been tested against a live pull request (no real GitHub environment was available during development). All integration tests use mocked GitHub API responses.

## GitHub Actions Integration

DocDrift ships a GitHub Actions workflow (`.github/workflows/docdrift.yml`) that automatically runs documentation-drift detection on every pull request.

### What it does

1. Triggers on `pull_request` events (opened, updated, reopened).
2. Fetches the pull request diff via the GitHub REST API.
3. Runs deterministic detection: reports renamed CLI flags whose old names appear in documentation.
4. Optionally runs AI semantic analysis (disabled by default — see below).
5. Writes a Markdown report to the **job summary** (visible on the Actions run page).
6. Optionally posts the report as a **pull request comment** (opt-in — see below).

### Required setup

No secrets or special configuration are required for basic usage. The workflow uses the automatically-provided `GITHUB_TOKEN` with only `contents: read` permission:

```yaml
permissions:
  contents: read
```

Enable the workflow by committing `.github/workflows/docdrift.yml` to your repository. It runs automatically on the next pull request.

### Enabling AI semantic analysis

GitHub-hosted runners cannot reach `localhost:11434`. AI analysis is therefore **disabled by default** in CI.

To enable it on a self-hosted runner with Ollama installed:

1. Set the `DOCDRIFT_AI_URL` repository variable (or environment variable) in your workflow:

```yaml
env:
  DOCDRIFT_AI_URL: "http://localhost:11434/api/chat"
  DOCDRIFT_AI_MODEL: "qwen3:4b"     # optional, default: qwen3:4b
  DOCDRIFT_AI_TIMEOUT: "60"         # optional, timeout in seconds
```

2. Run the workflow on a self-hosted runner that has Ollama serving the chosen model.

The workflow will automatically detect the non-empty `DOCDRIFT_AI_URL` and enable AI analysis.

### Enabling pull-request comments (opt-in)

By default, reports are written to the job summary only — no `pull-requests: write` permission is needed.

To post the report as a PR comment (with deduplication — old DocDrift comments are deleted before a new one is posted):

1. Add `pull-requests: write` to the workflow permissions block:

```yaml
permissions:
  contents: read
  pull-requests: write
```

2. Set the environment variable:

```yaml
env:
  DOCDRIFT_GITHUB_COMMENT: "true"
```

### Report format

The report contains two sections:

**Deterministic Findings** — verified, code-evidence-backed findings. Each entry includes:
- Finding type (`renamed_cli_flag`)
- Old and new flag names
- Documentation file and line number
- Outdated documentation text
- Code diff evidence
- Suggested correction

**AI Semantic Analysis** (when enabled) — unverified suggestions from the local model. Each entry includes:
- Description of the potential drift
- Evidence quote (with `✅ Verified verbatim` or `❌ Unverified` label)
- Confidence rating (`low` / `medium` / `high`)
- Suggested correction

> ⚠️ **AI findings are always labeled as unverified suggestions.** They may contain hallucinations. Always verify against the actual diff before acting.

### Permissions

| Permission | Why |
|---|---|
| `contents: read` | Check out repository and read documentation files |
| `pull-requests: write` | Post PR comments (opt-in only) |

No other permissions are requested. The `GITHUB_TOKEN` is never printed to logs.

### Limitations and known false-positive risks

- **AI false positives**: The AI model may flag behavior-preserving refactors (e.g., a loop rewritten as a list comprehension) as documentation drift. The improved prompt substantially reduces this, but local LLMs are non-deterministic. All AI findings require human review.
- **Deterministic scope**: Only CLI flag *renames* declared in source files are detected. Changed defaults, removed commands, and prose drift require AI analysis.
- **Diff size**: Diffs exceeding 512 KB are truncated before analysis. A warning appears in the report.
- **No live PR test**: The workflow logic has been tested with mocked GitHub API responses. A real PR test would require a live repository with GitHub Actions enabled.
- **GitHub Enterprise**: The API base URL can be overridden via `DOCDRIFT_GITHUB_API` environment variable if you use GitHub Enterprise Server.
- **AI in CI**: AI analysis requires a self-hosted runner with Ollama. It cannot be enabled on standard `ubuntu-latest` GitHub-hosted runners.

## Planned improvements

- Stronger evaluation of local-model semantic suggestions against known fixtures.
- Additional deterministic checks (removed options, changed defaults).
- Optional GitHub/CI reporting that points to the exact documentation sentence and code hunk.
