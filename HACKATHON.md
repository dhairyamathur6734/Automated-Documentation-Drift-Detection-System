# HACKATHON.md

## Team

- **Team name:** [Choose a team name]
- **Team members:**
  - Dhairya Mathur (solo participant)

## Problem Statement

**Problem 5 — Catch docs a change made wrong**

Build a CI check that identifies the exact documentation sentence made incorrect by a pull request, and links that sentence to the corresponding code change.

## Project

**Working name:** DocDrift

DocDrift will inspect code changes in a pull request, find related documentation, and report likely documentation drift with evidence. The initial scope focuses on high-confidence cases such as renamed CLI flags, changed configuration defaults, and removed options or symbols.

## AI Models

> Update this section to reflect only models actually used by the project. Do not claim a model that the code does not use.

- **Planned model family:** Qwen (open-weight).
- **Why:** An open-weight model can be used for semantic comparison between a code change and related documentation, while deterministic checks handle straightforward cases such as renamed identifiers and changed values.
- **Exact model/version and runtime:** To be filled in once the available model and access method are confirmed.
- **Current status:** Not integrated yet.

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
