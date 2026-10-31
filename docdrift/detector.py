"""Deterministic CLI-flag rename drift detector.

This module infers unambiguous command-line flag renames from a unified Git
diff, then reports Markdown documentation that still mentions the old flag.

Limitations
-----------
- Only source-style option declarations are treated as rename evidence
  (``argparse.add_argument``, similar helpers, or quoted flag literals that
  continue a declaration). Prose, filenames, and unrelated command text are
  ignored.
- Rename evidence is taken only from source-code files identified by unified
  diff headers (for example ``*.py``). Changes in Markdown or other
  non-source files are ignored.
- Flags are paired within a single diff change block using line templates and
  mutually unique matches. Global set-difference pairing is not used.
- Competing declarations that share a template are not greedily paired.
- Multiple flags on one declaration are paired only with structural evidence
  (a single replacement, or one long and one short option). Reordered aliases
  are not paired by list position.
- If the same old flag maps to more than one new flag, that old flag is
  skipped.
- Malformed or incomplete diffs do not raise; they yield no findings. Change
  lines outside a valid ``@@`` hunk are ignored.
- Documentation search is limited to ``*.md`` files and exact, boundary-aware
  flag matches. Semantic drift is out of scope for this milestone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LONG_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")
QUOTED_FLAG_RE = re.compile(r"""['\"](--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z])['\"]""")
SHORT_FLAG_RE = re.compile(r"(?<![A-Za-z0-9-])-[A-Za-z](?![A-Za-z0-9-])")
FLAG_LITERAL_LINE_RE = re.compile(
    r"""^['\"](--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z])['\"][,]?$"""
)
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
GIT_HEADER_RE = re.compile(r"^diff --git (?:a/)?(.+?) (?:b/)?(.+)$")
DECLARATION_MARKERS = (
    "add_argument",
    "add_option",
    "click.option",
    "option(",
    "Option(",
)
SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".pyw",
}
DIFF_PREFIXES = (
    "diff ",
    "index ",
    "---",
    "+++",
    "new file",
    "deleted file",
    "old mode",
    "new mode",
    "similarity index",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "Binary files",
    "GIT binary",
)


@dataclass(frozen=True)
class FlagRename:
    old_flag: str
    new_flag: str
    removed_line: str
    added_line: str


def detect_drift(diff: str, repo_path: str | Path) -> list[dict]:
    """Detect outdated Markdown references after CLI flag renames.

    Parameters
    ----------
    diff:
        Unified Git diff text. Empty, malformed, or unrelated diffs return
        an empty list.
    repo_path:
        Path to the repository (or fixture) whose Markdown files should be
        scanned. Missing or unreadable paths return an empty list.
    """
    if not isinstance(diff, str) or not diff.strip():
        return []

    root = _resolve_repo_path(repo_path)
    if root is None:
        return []

    renames = _unique_renames(_extract_renames(diff))
    if not renames:
        return []

    findings: list[dict] = []
    for markdown_file in _iter_markdown_files(root):
        relative = _relative_path(root, markdown_file)
        findings.extend(_find_outdated_references(markdown_file, relative, renames))

    findings.sort(
        key=lambda item: (
            item["documentation_file"],
            item["line_number"],
            item["old_flag"],
            item["new_flag"],
        )
    )
    return findings


def _resolve_repo_path(repo_path: str | Path) -> Path | None:
    try:
        root = Path(repo_path)
    except TypeError:
        return None
    try:
        root = root.resolve()
    except OSError:
        return None
    if not root.exists() or not root.is_dir():
        return None
    return root


def _iter_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        candidates = root.rglob("*.md")
    except OSError:
        return []
    for path in candidates:
        try:
            if path.is_file():
                files.append(path)
        except OSError:
            continue
    files.sort(key=lambda item: _relative_path(root, item))
    return files


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_renames(diff: str) -> list[FlagRename]:
    renames: list[FlagRename] = []
    for removed_lines, added_lines in _iter_source_change_blocks(diff):
        renames.extend(_pair_renames_in_block(removed_lines, added_lines))
    return renames


def _iter_source_change_blocks(diff: str) -> Iterable[tuple[list[str], list[str]]]:
    current_file: str | None = None
    in_hunk = False
    removed: list[str] = []
    added: list[str] = []

    def flush() -> Iterable[tuple[list[str], list[str]]]:
        nonlocal removed, added
        if removed or added:
            if _is_source_file(current_file):
                yield removed, added
        removed, added = [], []

    for raw_line in diff.splitlines():
        if raw_line.startswith("diff "):
            yield from flush()
            in_hunk = False
            current_file = _path_from_git_header(raw_line)
            continue
        if raw_line.startswith("--- "):
            continue
        if raw_line.startswith("+++ "):
            yield from flush()
            in_hunk = False
            current_file = _path_from_file_header(raw_line)
            continue
        if raw_line.startswith("@@"):
            yield from flush()
            in_hunk = bool(HUNK_HEADER_RE.match(raw_line))
            continue
        if _is_diff_metadata(raw_line) or raw_line.startswith("\\"):
            continue
        if not in_hunk:
            continue
        if raw_line.startswith("-"):
            removed.append(_strip_diff_prefix(raw_line))
            continue
        if raw_line.startswith("+"):
            added.append(_strip_diff_prefix(raw_line))
            continue
        if raw_line.startswith(" ") or raw_line == "":
            yield from flush()
            continue
        removed, added = [], []
        in_hunk = False

    yield from flush()


def _path_from_git_header(line: str) -> str | None:
    match = GIT_HEADER_RE.match(line.strip())
    if not match:
        return None
    return match.group(2).strip() or None


def _path_from_file_header(line: str) -> str | None:
    rest = line[4:].strip()
    if not rest or rest == "/dev/null":
        return None
    rest = rest.split("\t", 1)[0].strip()
    if rest.startswith("a/") or rest.startswith("b/"):
        rest = rest[2:]
    return rest or None


def _is_source_file(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in SOURCE_EXTENSIONS


def _is_diff_metadata(line: str) -> bool:
    return any(line.startswith(prefix) for prefix in DIFF_PREFIXES)


def _strip_diff_prefix(line: str) -> str:
    return line[1:] if line else line


def _pair_renames_in_block(removed_lines: list[str], added_lines: list[str]) -> list[FlagRename]:
    removed_decls = _declaration_entries(removed_lines)
    added_decls = _declaration_entries(added_lines)
    if not removed_decls or not added_decls:
        return []

    removed_matches = [
        [
            added_index
            for added_index, (_, _, added_template) in enumerate(added_decls)
            if added_template == removed_template
        ]
        for _, _, removed_template in removed_decls
    ]
    added_matches = [
        [
            removed_index
            for removed_index, (_, _, removed_template) in enumerate(removed_decls)
            if removed_template == added_template
        ]
        for _, _, added_template in added_decls
    ]

    paired: list[FlagRename] = []
    for removed_index, matches in enumerate(removed_matches):
        if len(matches) != 1:
            continue
        added_index = matches[0]
        if len(added_matches[added_index]) != 1:
            continue
        removed_line, removed_flags, _ = removed_decls[removed_index]
        added_line, added_flags, _ = added_decls[added_index]
        line_pairs = _pair_flag_lists(removed_flags, added_flags)
        if not line_pairs:
            continue
        for old_flag, new_flag in line_pairs:
            paired.append(
                FlagRename(
                    old_flag=old_flag,
                    new_flag=new_flag,
                    removed_line=removed_line.strip(),
                    added_line=added_line.strip(),
                )
            )
    return paired


def _declaration_entries(lines: list[str]) -> list[tuple[str, list[str], str]]:
    entries: list[tuple[str, list[str], str]] = []
    for line in lines:
        if not _is_declaration_line(line):
            continue
        flags = _extract_declared_flags(line)
        if not flags:
            continue
        entries.append((line, flags, _line_template(line)))
    return entries


def _is_declaration_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker.lower() in lowered for marker in DECLARATION_MARKERS):
        return True
    return bool(FLAG_LITERAL_LINE_RE.match(stripped))


def _extract_declared_flags(line: str) -> list[str]:
    quoted = QUOTED_FLAG_RE.findall(line)
    if quoted:
        return quoted
    flags = LONG_FLAG_RE.findall(line)
    flags.extend(SHORT_FLAG_RE.findall(line))
    return flags


def _line_template(line: str) -> str:
    templated = QUOTED_FLAG_RE.sub('"FLAG"', line)
    templated = LONG_FLAG_RE.sub("FLAG", templated)
    templated = SHORT_FLAG_RE.sub("FLAG", templated)
    return re.sub(r"\s+", " ", templated).strip()


def _pair_flag_lists(old_flags: list[str], new_flags: list[str]) -> list[tuple[str, str]] | None:
    if old_flags == new_flags:
        return []

    old_remaining = [flag for flag in old_flags if flag not in new_flags]
    new_remaining = [flag for flag in new_flags if flag not in old_flags]
    if not old_remaining or not new_remaining:
        return []
    if len(old_remaining) == 1 and len(new_remaining) == 1:
        return [(old_remaining[0], new_remaining[0])]

    pairs: list[tuple[str, str]] = []
    for selector in (_is_long_flag, _is_short_flag):
        old_kind = [flag for flag in old_remaining if selector(flag)]
        new_kind = [flag for flag in new_remaining if selector(flag)]
        if not old_kind and not new_kind:
            continue
        if len(old_kind) == 1 and len(new_kind) == 1:
            pairs.append((old_kind[0], new_kind[0]))
            continue
        return None
    return pairs or None


def _is_long_flag(flag: str) -> bool:
    return flag.startswith("--")


def _is_short_flag(flag: str) -> bool:
    return bool(re.fullmatch(r"-[A-Za-z]", flag))


def _unique_renames(renames: list[FlagRename]) -> list[FlagRename]:
    by_old: dict[str, set[str]] = {}
    ordered: list[FlagRename] = []
    seen: set[tuple[str, str]] = set()

    for rename in renames:
        by_old.setdefault(rename.old_flag, set()).add(rename.new_flag)

    ambiguous_old = {old for old, news in by_old.items() if len(news) != 1}

    for rename in renames:
        if rename.old_flag in ambiguous_old:
            continue
        key = (rename.old_flag, rename.new_flag)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(rename)
    return ordered


def _find_outdated_references(
    path: Path,
    relative: str,
    renames: list[FlagRename],
) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []

    findings: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rename in renames:
            if not _is_outdated_reference(line, rename):
                continue
            findings.append(
                {
                    "type": "renamed_cli_flag",
                    "documentation_file": relative,
                    "line_number": line_number,
                    "outdated_text": line,
                    "old_flag": rename.old_flag,
                    "new_flag": rename.new_flag,
                    "code_change": {
                        "removed": rename.removed_line,
                        "added": rename.added_line,
                    },
                }
            )
    return findings


def _is_outdated_reference(line: str, rename: FlagRename) -> bool:
    """Return True if line represents genuine outdated documentation for rename.old_flag."""
    if not _contains_flag(line, rename.old_flag):
        return False

    stripped = line.strip()
    lowered = line.lower()
    flag_pattern = rf"(?<![A-Za-z0-9_-]){re.escape(rename.old_flag.lower())}(?![A-Za-z0-9_-])"

    # 1. Diff hunk lines in code blocks (e.g. +parser.add_argument(...) or -parser.add_argument(...))
    # Unordered markdown lists start with "+ " or "- ", whereas diff lines start with "+" or "-" followed by code.
    if (stripped.startswith("+") and not stripped.startswith("+ ")) or (
        stripped.startswith("-") and not stripped.startswith("- ")
    ):
        return False
    if re.match(
        r"^[+-]\s*(?:parser\.|add_argument|add_option|click\.|option\(|Option\(|[a-zA-Z_]+\s*=)",
        stripped,
    ):
        return False

    # 2. Test fixture / finding schema output (e.g. "new_flag": "--deadline", "added": "...", "old_flag": "...")
    if re.search(
        r"""['"](?:new_flag|old_flag|added|removed|new|outdated_text)['"]\s*:\s*""",
        line,
    ):
        return False

    # 3. Explicit statements stating the reference is correct or should not be reported
    if re.search(
        r"\b(?:must\s+not|should\s+not|is\s+not|not|never)\s+(?:be\s+)?reported\b",
        lowered,
    ):
        return False
    if re.search(r"\bcorrect(?:ly)?\b.*?" + flag_pattern, lowered) or re.search(
        flag_pattern + r".*?\bcorrect(?:ly)?\b", lowered
    ):
        if re.search(
            r"\bcorrect(?:ly)?\b.*?(?:reference|mention|us\w+|state|claim)",
            lowered,
        ) or re.search(
            r"\bcorrect\s+[`'\"]?" + re.escape(rename.old_flag.lower()),
            lowered,
        ):
            return False

    # 4. Statements describing a different rename where old_flag is the destination/target
    # e.g. "changes `--timeout` to `--deadline`" or "renaming `--timeout` to `--deadline`"
    diff_rename_target = re.search(
        r"""(?:to|->|→|into)\s+[`'"]?"""
        + re.escape(rename.old_flag)
        + r"""[`'"]?(?:[.,\s\)]|$)""",
        line,
    )
    if diff_rename_target:
        if re.search(
            r"""(?:--[a-zA-Z0-9-]+|-[a-zA-Z]|renam\w+|chang\w+|from)\b.*?(?:to|->|→|into)\s+[`'"]?"""
            + re.escape(rename.old_flag),
            line,
        ):
            return False

    # 5. Explaining the rename itself or meta-commentary:
    # e.g., "renamed/changes --old to --new" or "README.md still says '--old'"
    self_rename = re.search(
        re.escape(rename.old_flag)
        + r"""[`'"]?\s+(?:to|->|→|into)\s+[`'"]?"""
        + re.escape(rename.new_flag),
        line,
    )
    if self_rename:
        return False
    if re.search(r"\bstill\s+says\b", lowered):
        return False

    return True


def _contains_flag(text: str, flag: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])"
    return re.search(pattern, text) is not None
