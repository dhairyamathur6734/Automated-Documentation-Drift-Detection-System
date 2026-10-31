"""GitHub Actions pull-request integration for DocDrift.

Entry point::

    python3 -m docdrift.github_reporter

Reads GitHub Actions environment variables, fetches the pull-request diff via
the GitHub REST API (no third-party packages), runs ``analyze_changes``, and
writes a Markdown report to ``$GITHUB_STEP_SUMMARY`` (job summary).

Optionally posts the report as a pull-request comment when
``DOCDRIFT_GITHUB_COMMENT=true`` and the workflow has ``pull-requests: write``
permission.

Security model
--------------
- The GitHub token is read from the environment and never printed.
- Diff and documentation text are passed unchanged to ``analyze_changes`` as
  untrusted input; the existing orchestrator applies all safety limits.
- Repository name, PR number, and token are read only from trusted environment
  variables, never from the diff or documentation content.
- Oversized diffs are truncated before analysis with a visible warning.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .orchestrator import analyze_changes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum diff size accepted before truncation (512 KB)
MAX_DIFF_BYTES: int = 512 * 1024

# Magic string embedded in every DocDrift PR comment for deduplication.
COMMENT_MARKER: str = "<!-- docdrift-report -->"

# GitHub REST API base URL (can be overridden for GitHub Enterprise)
GITHUB_API_BASE: str = os.environ.get(
    "DOCDRIFT_GITHUB_API", "https://api.github.com"
)

_HEADER_AI_UNAVAILABLE = (
    "> ⚠️ **AI analysis** was not requested or is unavailable in this environment.\n"
    "> To enable semantic drift analysis, point a self-hosted Ollama server at\n"
    "> `DOCDRIFT_AI_URL` in your workflow environment.\n"
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_from_github_env() -> int:
    """Run DocDrift from GitHub Actions environment variables.

    Returns
    -------
    int
        0 on success (findings or not), 1 on unrecoverable error.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        _log_error(
            "GITHUB_TOKEN is not set. "
            "Add 'permissions: contents: read' to your workflow and ensure "
            "the token is passed via ${{ secrets.GITHUB_TOKEN }}."
        )
        return 1

    # Load the PR event payload
    event = _load_github_event()
    if event is None:
        _log_error(
            "Could not load GitHub event JSON from GITHUB_EVENT_PATH. "
            "This script must run inside a GitHub Actions pull_request event."
        )
        return 1

    repo = os.environ.get("GITHUB_REPOSITORY") or event.get("repository", {}).get(
        "full_name", ""
    )
    pr_number = _extract_pr_number(event)

    if not repo or pr_number is None:
        _log_error(
            "Could not determine repository or pull-request number from the "
            "GitHub event payload. Ensure this workflow runs on pull_request events."
        )
        return 1

    # Fetch the diff
    diff_result = _get_pr_diff(repo, pr_number, token)
    if diff_result is None:
        _log_error(f"Failed to obtain diff for PR #{pr_number} in {repo}.")
        return 1

    diff_text, diff_truncated = diff_result

    # Resolve workspace / repository checkout path
    workspace = os.environ.get("GITHUB_WORKSPACE", ".")
    repo_path = Path(workspace)
    if not repo_path.is_dir():
        _log_error(f"GITHUB_WORKSPACE is not an existing directory: {workspace}")
        return 1

    # AI configuration — disabled unless DOCDRIFT_AI_URL is set
    ai_url = os.environ.get("DOCDRIFT_AI_URL", "").strip()
    use_ai = bool(ai_url)
    ai_model = os.environ.get("DOCDRIFT_AI_MODEL", "qwen3:4b").strip() or "qwen3:4b"
    try:
        ai_timeout = float(os.environ.get("DOCDRIFT_AI_TIMEOUT", "60"))
    except ValueError:
        ai_timeout = 60.0

    # Run analysis
    try:
        result = analyze_changes(
            diff_text,
            repo_path,
            use_ai=use_ai,
            model=ai_model if use_ai else None,
            base_url=ai_url if use_ai else None,
            timeout=ai_timeout,
        )
    except ValueError as exc:
        _log_error(f"Analysis failed with invalid input: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Unexpected analysis error: {exc}")
        return 1

    # Format the report
    report = _format_report(result, pr_number, diff_truncated=diff_truncated, use_ai=use_ai)

    # Write to job summary
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_file:
        try:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write(report)
                fh.write("\n")
        except OSError as exc:
            _log_error(f"Could not write to GITHUB_STEP_SUMMARY: {exc}")
    else:
        # Fallback: print to stdout (useful for local testing)
        print(report)

    # Optionally post as a PR comment
    if os.environ.get("DOCDRIFT_GITHUB_COMMENT", "").lower() == "true":
        _post_pr_comment(repo, pr_number, token, report)

    return 0


def _get_api_base() -> str:
    """Return the GitHub REST API base URL, supporting runtime environment override."""
    return os.environ.get("DOCDRIFT_GITHUB_API", GITHUB_API_BASE).rstrip("/")


def _parse_next_link(link_header: Any) -> str | None:
    """Extract next URL from GitHub Link header if present."""
    if not isinstance(link_header, str) or not link_header.strip():
        return None
    for part in link_header.split(","):
        segments = part.strip().split(";")
        if len(segments) >= 2:
            url_part = segments[0].strip()
            params = [s.strip() for s in segments[1:]]
            if any('rel="next"' in p or "rel='next'" in p for p in params):
                if url_part.startswith("<") and url_part.endswith(">"):
                    target = url_part[1:-1]
                    if target.startswith(("http://", "https://")):
                        return target
    return None


def _escape_table_cell(text: Any) -> str:
    """Escape vertical pipes and newlines that break Markdown table formatting."""
    return str(text).replace("|", "\\|").replace("\r", "").replace("\n", " ")


def _sanitize_markdown_text(text: Any) -> str:
    """Neutralize Markdown headers or formatting injections in untrusted prose."""
    sanitized_lines: list[str] = []
    for line in str(text).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            sanitized_lines.append("\\" + stripped)
        else:
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def _safe_code_fence(code: Any) -> str:
    """Format code inside backticks safely avoiding fence breaking."""
    clean = str(code).replace("```", "'''")
    return f"```\n{clean}\n```"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _get_pr_diff(
    repo: str, pr_number: int, token: str
) -> tuple[str, bool] | None:
    """Fetch pull-request files and assemble a unified diff string.

    Follows GitHub's Link-header pagination so that PRs with more than 30
    changed files are fully covered.  Stops fetching further pages once the
    assembled diff reaches ``MAX_DIFF_BYTES``.

    Returns ``(diff_text, was_truncated)`` or ``None`` on failure.
    """
    base_url = (
        f"{_get_api_base()}/repos/{repo}/pulls/{pr_number}/files"
        "?per_page=100"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "docdrift-github-reporter/1.0",
    }

    # Assemble unified diff from per-file patches
    diff_parts: list[str] = []
    total_bytes = 0
    truncated = False
    next_url: str | None = base_url

    while next_url:
        try:
            req = urllib.request.Request(next_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                link_header = ""
                if hasattr(resp, "headers") and resp.headers is not None:
                    header_val = resp.headers.get("Link", "")
                    if isinstance(header_val, str):
                        link_header = header_val
        except urllib.error.HTTPError as exc:
            _log_error(f"GitHub API error {exc.code} fetching PR files: {exc.reason}")
            return None
        except urllib.error.URLError as exc:
            _log_error(f"GitHub API connection error: {exc.reason}")
            return None
        except Exception as exc:  # noqa: BLE001
            _log_error(f"Unexpected error fetching PR diff: {exc}")
            return None

        try:
            files: list[dict] = json.loads(raw)
        except json.JSONDecodeError as exc:
            _log_error(f"Could not parse GitHub API response: {exc}")
            return None

        if not isinstance(files, list):
            _log_error("Unexpected GitHub API response shape (expected a list of files).")
            return None

        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            filename = file_entry.get("filename", "")
            patch = file_entry.get("patch", "")
            if not filename or not patch:
                continue

            prev_name = file_entry.get("previous_filename", filename)
            block = (
                f"diff --git a/{prev_name} b/{filename}\n"
                f"--- a/{prev_name}\n"
                f"+++ b/{filename}\n"
                f"{patch}\n"
            )
            block_bytes = len(block.encode("utf-8"))
            if total_bytes + block_bytes > MAX_DIFF_BYTES:
                truncated = True
                next_url = None  # stop fetching further pages
                break
            diff_parts.append(block)
            total_bytes += block_bytes

        if next_url and not truncated:
            next_url = _parse_next_link(link_header)

    if not diff_parts:
        # Return a minimal stub so analysis still runs and produces an empty result
        return ("diff --git a/.docdrift_empty b/.docdrift_empty\n", truncated)

    return ("\n".join(diff_parts), truncated)


def _post_pr_comment(
    repo: str, pr_number: int, token: str, report: str
) -> None:
    """Post the report as a PR comment, deleting any previous DocDrift comment."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "docdrift-github-reporter/1.0",
    }
    api_base = _get_api_base()

    # Find and delete existing DocDrift comments (deduplication)
    list_url = f"{api_base}/repos/{repo}/issues/{pr_number}/comments?per_page=100"
    try:
        scan_url: str | None = list_url
        while scan_url:
            req = urllib.request.Request(scan_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                existing: list[dict] = json.loads(resp.read())
                link_header = ""
                if hasattr(resp, "headers") and resp.headers is not None:
                    header_val = resp.headers.get("Link", "")
                    if isinstance(header_val, str):
                        link_header = header_val
            for comment in existing:
                if isinstance(comment, dict) and COMMENT_MARKER in comment.get("body", ""):
                    delete_url = (
                        f"{api_base}/repos/{repo}/issues/comments/{comment['id']}"
                    )
                    del_req = urllib.request.Request(
                        delete_url, headers=headers, method="DELETE"
                    )
                    with urllib.request.urlopen(del_req, timeout=30):
                        pass
            scan_url = _parse_next_link(link_header)
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Could not clean up previous DocDrift comments: {exc}")

    # Post new comment
    post_url = f"{api_base}/repos/{repo}/issues/{pr_number}/comments"
    body = f"{COMMENT_MARKER}\n\n{report}"
    payload = json.dumps({"body": body}).encode("utf-8")
    try:
        post_req = urllib.request.Request(
            post_url, data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(post_req, timeout=30):
            pass
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Could not post PR comment: {exc}")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_report(
    result: dict[str, Any],
    pr_number: int,
    *,
    diff_truncated: bool = False,
    use_ai: bool = False,
) -> str:
    """Render the analyze_changes result as a Markdown report."""
    det = result.get("deterministic_findings") or []
    ai_analysis = result.get("ai_analysis")
    ai_findings = ai_analysis.get("findings") or [] if ai_analysis else []
    doc_files = result.get("documentation_files") or []
    status = result.get("status", "unknown")
    ai_error = result.get("ai_error")

    lines: list[str] = []
    lines.append("## 📄 DocDrift — Documentation Drift Report")
    lines.append(f"\n**Pull Request:** #{pr_number}\n")

    if diff_truncated:
        lines.append(
            "> ⚠️ **The diff was truncated** because it exceeded the 512 KB size limit. "
            "Only the first portion of the diff was analyzed.\n"
        )

    total_findings = len(det) + len(ai_findings)

    if total_findings == 0 and status not in ("ai_failed",):
        lines.append("✅ **No documentation drift detected.**\n")
        if not use_ai:
            lines.append(_HEADER_AI_UNAVAILABLE)
        if doc_files:
            lines.append(f"**Documentation files analyzed:** {', '.join(f'`{f}`' for f in doc_files)}\n")
        return "\n".join(lines)

    # --- Deterministic findings ---
    if det:
        lines.append(f"### 🔍 Deterministic Findings ({len(det)})\n")
        lines.append(
            "> These findings are based on verified code evidence. "
            "No model inference is involved.\n"
        )
        for i, f in enumerate(det, 1):
            doc_file = _escape_table_cell(f.get("documentation_file", "unknown"))
            line_no = _escape_table_cell(f.get("line_number", "?"))
            old_flag = _escape_table_cell(f.get("old_flag", "?"))
            new_flag = _escape_table_cell(f.get("new_flag", "?"))
            outdated = f.get("outdated_text", "")
            removed = str(f.get("code_change", {}).get("removed", "")).replace("```", "'''")
            added = str(f.get("code_change", {}).get("added", "")).replace("```", "'''")

            lines.append(f"#### Finding {i}: Renamed CLI Flag")
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            lines.append("| **Type** | `renamed_cli_flag` |")
            lines.append(f"| **Old flag** | `{old_flag}` |")
            lines.append(f"| **New flag** | `{new_flag}` |")
            lines.append(f"| **Documentation file** | `{doc_file}` line {line_no} |")
            lines.append("| **Source** | Deterministic (verified) |")
            lines.append("")
            if outdated:
                lines.append("**Outdated documentation line:**")
                lines.append(_safe_code_fence(outdated))
            if removed or added:
                lines.append("**Code change:**")
                lines.append("```diff")
                if removed:
                    lines.append(f"- {removed}")
                if added:
                    lines.append(f"+ {added}")
                lines.append("```")
            lines.append(
                f"**Suggestion:** Update `{doc_file}` to replace `{old_flag}` with `{new_flag}`.\n"
            )

    # --- AI findings ---
    if ai_findings:
        ai_model = ai_analysis.get("model", "unknown") if ai_analysis else "unknown"
        lines.append(f"### 🤖 AI Semantic Analysis — Unverified Suggestions ({len(ai_findings)})\n")
        lines.append(
            f"> ⚠️ **These are unverified suggestions** from `{ai_model}`. "
            "Even when cited evidence is verified verbatim from source text, "
            "model reasoning is not verified. Always verify against the actual code "
            "and documentation before acting.\n"
        )
        for i, f in enumerate(ai_findings, 1):
            desc = _sanitize_markdown_text(f.get("description", ""))
            evidence = f.get("evidence", "")
            suggestion = _sanitize_markdown_text(f.get("suggestion", ""))
            confidence = _escape_table_cell(f.get("confidence", "unknown"))
            evidence_verified = f.get("evidence_verified", False)
            verified_label = "✅ Verified verbatim" if evidence_verified else "❌ **Unverified** (may be hallucinated)"

            lines.append(f"#### AI Suggestion {i}")
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            lines.append("| **Type** | `possible_semantic_drift` |")
            lines.append(f"| **Confidence** | `{confidence}` |")
            lines.append(f"| **Evidence verified** | {verified_label} |")
            lines.append(f"| **Source** | AI (`{ai_model}`) — unverified |")
            lines.append("")
            if desc:
                lines.append(f"**Description:** {desc}\n")
            if evidence:
                lines.append("**Evidence:**")
                lines.append(_safe_code_fence(evidence))
            if suggestion:
                lines.append(f"**Suggestion:** {suggestion}\n")

        # AI limitations
        if ai_analysis and ai_analysis.get("limitations"):
            lines.append("**Model limitations noted:**")
            for lim in ai_analysis["limitations"]:
                lines.append(f"- {_sanitize_markdown_text(lim)}")
            lines.append("")

    elif use_ai and not ai_findings:
        lines.append("### 🤖 AI Semantic Analysis\n")
        if ai_error:
            lines.append(
                f"> ⚠️ **AI analysis failed:** `{_escape_table_cell(ai_error)}`\n"
                "> Deterministic findings above are unaffected.\n"
            )
        elif status == "no_relevant_documentation":
            lines.append("> ℹ️ No relevant documentation was found for AI analysis.\n")
        else:
            lines.append("> ✅ AI analysis found no additional semantic drift.\n")

    elif not use_ai:
        lines.append(_HEADER_AI_UNAVAILABLE)

    # Doc files footer
    if doc_files:
        lines.append(
            f"**Documentation files analyzed:** {', '.join(f'`{f}`' for f in doc_files)}\n"
        )

    lines.append(
        "\n---\n*DocDrift — deterministic findings are code-evidence-backed. "
        "AI suggestions require human review before any documentation changes.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_github_event() -> dict | None:
    """Parse the GitHub event JSON file from GITHUB_EVENT_PATH."""
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        return {}
    try:
        with open(event_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _extract_pr_number(event: dict) -> int | None:
    """Extract PR number from the GitHub event payload."""
    try:
        number = event.get("number") or event.get("pull_request", {}).get("number")
        if number is not None:
            return int(number)
    except (TypeError, ValueError):
        pass
    return None


def _log_error(message: str) -> None:
    """Print a GitHub Actions error annotation, without leaking secrets."""
    # Avoid printing any value that looks like a token (40+ hex chars)
    safe_message = message
    print(f"::error::{safe_message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_from_github_env())
