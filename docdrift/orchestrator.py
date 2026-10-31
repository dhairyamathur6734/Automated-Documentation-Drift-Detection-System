"""Orchestration API for DocDrift documentation drift detection.

Combines deterministic CLI flag rename detection with optional AI-powered
semantic drift analysis against relevant repository documentation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .ai_detector import analyze_semantic_drift
from .detector import detect_drift

MAX_FILE_BYTES = 64 * 1024  # 64 KB per documentation file
MAX_TOTAL_DOC_BYTES = 96 * 1024  # 96 KB total documentation payload

FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*|-[A-Za-z]")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
DIFF_GIT_RE = re.compile(r"^diff --git (?:a/)?(\S+) (?:b/)?(\S+)")
DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)?(\S+)")
STOP_WORDS = {
    "and", "class", "def", "elif", "else", "except", "false", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "none",
    "not", "or", "pass", "raise", "return", "self", "true", "try", "while",
    "with", "yield", "diff", "git", "index", "null", "dev",
}


def analyze_changes(
    diff: str,
    repo_path: str | Path,
    *,
    use_ai: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Analyze code changes against repository documentation.

    Runs deterministic detection for renamed CLI flags, and optionally
    runs AI semantic analysis using local Ollama against relevant documentation.

    Parameters
    ----------
    diff:
        Unified Git diff text describing code changes. Must be a non-empty string.
    repo_path:
        Path to the repository directory to inspect.
    use_ai:
        If True, queries the local Ollama model for semantic drift in relevant
        documentation. If False (default), runs offline without calling Ollama.
    model:
        Ollama model name (e.g. ``qwen3:4b``).
    base_url:
        Ollama endpoint or server URL.
    timeout:
        HTTP timeout in seconds for AI analysis.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - ``deterministic_findings``: list of verified deterministic findings.
        - ``ai_analysis``: model response dictionary (containing ``findings``,
          ``limitations``, ``model``, and ``status``), or None.
        - ``documentation_files``: list of relative paths of analyzed doc files.
        - ``status``: execution status string (``completed``,
          ``no_relevant_documentation``, or ``ai_failed``).
        - ``ai_error``: error message string if AI analysis failed, else None.

    Raises
    ------
    ValueError
        If ``diff`` is empty/whitespace or ``repo_path`` is not an existing directory.
    """
    if not isinstance(diff, str) or not diff.strip():
        raise ValueError("diff must be a non-empty string")

    root = _resolve_repo_dir(repo_path)
    if not isinstance(use_ai, bool):
        raise ValueError("use_ai must be a boolean")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be a positive number")

    deterministic_findings = detect_drift(diff, root)

    if not use_ai:
        return {
            "deterministic_findings": deterministic_findings,
            "ai_analysis": None,
            "documentation_files": [],
            "status": "completed",
            "ai_error": None,
        }

    selected_paths, documentation_text = _select_relevant_docs(
        diff, root, deterministic_findings
    )

    if not selected_paths or not documentation_text.strip():
        return {
            "deterministic_findings": deterministic_findings,
            "ai_analysis": None,
            "documentation_files": [],
            "status": "no_relevant_documentation",
            "ai_error": None,
        }

    try:
        ai_result = analyze_semantic_drift(
            diff,
            documentation_text,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        return {
            "deterministic_findings": deterministic_findings,
            "ai_analysis": ai_result,
            "documentation_files": selected_paths,
            "status": "completed",
            "ai_error": None,
        }
    except Exception as exc:
        return {
            "deterministic_findings": deterministic_findings,
            "ai_analysis": None,
            "documentation_files": selected_paths,
            "status": "ai_failed",
            "ai_error": str(exc),
        }


def _resolve_repo_dir(repo_path: str | Path) -> Path:
    if repo_path is None or not isinstance(repo_path, (str, Path)):
        raise ValueError("repo_path must be a valid path string or Path object")
    try:
        resolved = Path(repo_path).resolve()
    except (TypeError, OSError) as exc:
        raise ValueError(f"Invalid repo_path: {repo_path}") from exc

    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"repo_path must be an existing directory: {repo_path}")
    return resolved


def _add_file_stem(candidate: str, file_stems: set[str]) -> None:
    candidate = candidate.strip()
    if candidate and candidate != "/dev/null":
        p = Path(candidate)
        if p.name:
            file_stems.add(p.name.lower())
            file_stems.add(p.stem.lower())


def _extract_diff_signals(diff: str) -> tuple[set[str], set[str], set[str]]:
    file_stems: set[str] = set()
    flags: set[str] = set()
    identifiers: set[str] = set()

    for line in diff.splitlines():
        git_match = DIFF_GIT_RE.match(line)
        if git_match:
            for candidate in git_match.groups():
                _add_file_stem(candidate, file_stems)
            continue

        file_match = DIFF_FILE_RE.match(line)
        if file_match:
            _add_file_stem(file_match.group(1), file_stems)
            continue

        if line.startswith(("-", "+")):
            line_text = line[1:].strip()
            for flag in FLAG_RE.findall(line_text):
                flags.add(flag.lower())
            for ident in IDENTIFIER_RE.findall(line_text):
                lowered = ident.lower()
                if lowered not in STOP_WORDS:
                    identifiers.add(lowered)

    return file_stems, flags, identifiers


def _score_document(
    rel_path: str,
    content: str,
    file_stems: set[str],
    flags: set[str],
    identifiers: set[str],
    deterministic_files: set[str],
) -> int:
    score = 0
    rel_lowered = rel_path.lower()
    content_lowered = content.lower()

    if rel_path in deterministic_files:
        score += 100

    doc_path_obj = Path(rel_lowered)
    if doc_path_obj.name in file_stems or doc_path_obj.stem in file_stems:
        score += 40

    for flag in flags:
        if flag in content_lowered:
            score += 20

    matched_idents = 0
    for ident in identifiers:
        if ident in content_lowered:
            matched_idents += 1
            if matched_idents <= 5:
                score += 5

    if doc_path_obj.name in ("readme.md", "index.md") and score > 0:
        score += 15

    return score


def _select_relevant_docs(
    diff: str,
    root: Path,
    deterministic_findings: list[dict],
) -> tuple[list[str], str]:
    deterministic_files = {
        item["documentation_file"]
        for item in deterministic_findings
        if isinstance(item, dict) and "documentation_file" in item
    }
    file_stems, flags, identifiers = _extract_diff_signals(diff)

    candidates: list[tuple[int, str, str]] = []

    try:
        markdown_files = sorted(root.rglob("*.md"))
    except OSError:
        return [], ""

    seen_resolved: set[Path] = set()

    for path in markdown_files:
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue

        try:
            if not path.is_file():
                continue
            resolved_path = path.resolve()
            resolved_parts = resolved_path.relative_to(root).parts
        except (ValueError, OSError):
            continue

        if any(part.startswith(".") for part in resolved_parts):
            continue

        if resolved_path in seen_resolved:
            continue
        seen_resolved.add(resolved_path)

        rel_path = path.relative_to(root).as_posix()

        try:
            with open(path, "rb") as f:
                raw_bytes = f.read(MAX_FILE_BYTES + 1)
        except (OSError, PermissionError):
            continue

        was_truncated = len(raw_bytes) > MAX_FILE_BYTES
        try:
            text = raw_bytes[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
        except UnicodeError:
            continue

        score = _score_document(
            rel_path,
            text,
            file_stems,
            flags,
            identifiers,
            deterministic_files,
        )

        if score > 0:
            if was_truncated:
                text += "\n\n... [content truncated due to size limit] ...\n"
            candidates.append((score, rel_path, text))

    if not candidates:
        return [], ""

    candidates.sort(key=lambda x: (-x[0], x[1]))

    selected_paths: list[str] = []
    combined_sections: list[str] = []
    total_bytes = 0

    for _, rel_path, text in candidates:
        text_bytes = len(text.encode("utf-8"))
        if total_bytes + text_bytes > MAX_TOTAL_DOC_BYTES and combined_sections:
            break
        selected_paths.append(rel_path)
        combined_sections.append(f"# Documentation File: {rel_path}\n\n{text}")
        total_bytes += text_bytes

    combined_text = "\n\n---\n\n".join(combined_sections)
    return selected_paths, combined_text
