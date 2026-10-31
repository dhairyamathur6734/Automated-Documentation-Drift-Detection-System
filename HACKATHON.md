# [HACKATHON.md](http://HACKATHON.md)

## Team

- **Team name:** Musical Box
- **Team members:**
  - Dhairya Mathur (solo participant)

## Problem Statement

**Problem 5 — Catch docs a change made wrong**

Build a CI check that identifies the exact documentation sentence made incorrect by a pull request, and links that sentence to the corresponding code change.

## Selected problem

Problem 5. This milestone implements the first reliable detection engine for that problem: renamed CLI flags found in a unified Git diff and reported against Markdown documentation.

## Project

**Working name:** DocDrift

DocDrift will inspect code changes in a pull request, find related documentation, and report likely documentation drift with evidence. The initial scope focuses on high-confidence cases such as renamed CLI flags, changed configuration defaults, and removed options or symbols.

## Proposed solution

Start with a deterministic parser rather than a model. Given a unified diff, DocDrift infers old → new CLI flag pairs from source declarations (for example `argparse.add_argument`), then searches Markdown files for leftover references to the old flag. Each finding includes the documentation path, one-based line number, original sentence, both flag names, and the removed/added code lines.

An experimental local Ollama helper (`analyze_semantic_drift`) can suggest semantic drift for a human to review. It is optional and independent of the deterministic detector.

## Target users

Maintainers and contributors who ship CLIs and want pull-request checks that catch docs still describing the previous flag names.

## Current MVP scope

In scope:

- Parse unified Git diffs without executing repository code.
- Infer unambiguous CLI flag renames, including long options and short options.
- Scan Markdown files recursively under a supplied repository path.
- Return structured Python dictionaries for outdated old-flag references.
- Optionally ask a local Ollama model for unverified semantic-drift suggestions.

Out of scope for this milestone:

- Cloud AI APIs
- GitHub App, frontend, or database
- Generic documentation linting
- Hardcoded sample findings

## Technical architecture

```text
unified diff  →  hunk / change-block parser
              →  declaration-aware flag extraction
              →  conservative rename pairing
repository    →  recursive Markdown scan
              →  boundary-aware old-flag matching
              →  structured findings
```

Public APIs: `docdrift.detector.detect_drift(diff, repo_path) -> list[dict]` and optional `docdrift.ai_detector.analyze_semantic_drift(diff, documentation) -> dict`.

The implementation uses only the Python standard library.

## Demonstration scenario

`examples/sample_repo/` contains current code that defines `--deadline` and a README that still mentions `--timeout` while also correctly mentioning `--deadline`.

A unified diff renaming `--timeout` to `--deadline` should produce exactly one finding: README.md line 3 (`Use` --timeout `to stop slow jobs.`). The `--deadline` sentence must not be reported.

## Current limitations

- Only straightforward, evidence-backed flag renames are reported.
- Ambiguous multi-flag change blocks are skipped to avoid false pairings.
- Semantic suggestions from the local model are unverified and require human review.
- No CI integration is bundled in this milestone.

## Planned future improvements

- Evaluate local-model semantic suggestions more thoroughly.
- Expand deterministic checks to removed options and changed defaults.
- Add a CI-friendly reporter that points to the exact documentation sentence and code change.

## AI Models

> Update this section to reflect only models actually used by the project. Do not claim a model that the code does not use.

- **Model family:** Qwen (open-weight).
- **Why:** An open-weight model can be used for semantic comparison between a code change and related documentation, while deterministic checks handle straightforward cases such as renamed identifiers and changed values.
- **Exact model/version and runtime:** `qwen3:4b` via local Ollama (`http://localhost:11434/api/chat` by default). Configurable with `DOCDRIFT_AI_MODEL` and `DOCDRIFT_OLLAMA_URL`.
- **Current status:** Integrated only as an experimental, optional helper. Findings are unverified suggestions. The deterministic detector does not call this model.

The project will use only an eligible open-weight model for model-powered functionality. Closed-model APIs such as OpenAI, Claude, and Gemini will not be used by the product.

## AI Coding Assistants

Record the coding assistants actually used for each pull request in that PR's description, as required by the hackathon rules.

## Initial Implementation Plan

1. Parse the pull-request diff and identify relevant code changes.
2. Search documentation files for references related to those changes.
3. Detect likely mismatches, starting with renamed flags, changed defaults, and removed options.
4. Use the open-weight model for cases that require semantic interpretation.
5. Verify findings and report the exact documentation file/line and corresponding code change.
6. Add tests and a reproducible example under `examples/`.

## License

MIT
