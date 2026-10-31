"""Tests for docdrift.github_reporter.

All tests mock urllib.request.urlopen and GitHub environment variables.
No live GitHub credentials or Ollama access is required.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from docdrift.github_reporter import (
    COMMENT_MARKER,
    MAX_DIFF_BYTES,
    _escape_table_cell,
    _extract_pr_number,
    _format_report,
    _get_api_base,
    _get_pr_diff,
    _load_github_event,
    _parse_next_link,
    _post_pr_comment,
    _safe_code_fence,
    _sanitize_markdown_text,
    run_from_github_env,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLAG_RENAME_RESULT = {
    "deterministic_findings": [
        {
            "type": "renamed_cli_flag",
            "documentation_file": "README.md",
            "line_number": 3,
            "outdated_text": "Use `--timeout` to stop slow jobs.",
            "old_flag": "--timeout",
            "new_flag": "--deadline",
            "code_change": {
                "removed": 'parser.add_argument("--timeout", help="Execution time")',
                "added": 'parser.add_argument("--deadline", help="Execution time")',
            },
        }
    ],
    "ai_analysis": None,
    "documentation_files": [],
    "status": "completed",
    "ai_error": None,
}

_NO_FINDINGS_RESULT = {
    "deterministic_findings": [],
    "ai_analysis": None,
    "documentation_files": [],
    "status": "completed",
    "ai_error": None,
}

_AI_RESULT = {
    "deterministic_findings": [],
    "ai_analysis": {
        "findings": [
            {
                "type": "possible_semantic_drift",
                "description": "Default timeout changed from 30 to 60 seconds.",
                "evidence": "DEFAULT_TIMEOUT = 30\n+DEFAULT_TIMEOUT = 60",
                "suggestion": "Update docs to say 60 seconds.",
                "confidence": "high",
                "evidence_verified": True,
            }
        ],
        "limitations": ["Assumes DEFAULT_TIMEOUT maps to documented timeout."],
        "model": "qwen3:4b",
        "status": "unverified_ai_suggestions",
    },
    "documentation_files": ["docs/config.md"],
    "status": "completed",
    "ai_error": None,
}

_AI_FAILED_RESULT = {
    "deterministic_findings": [],
    "ai_analysis": None,
    "documentation_files": [],
    "status": "ai_failed",
    "ai_error": "Timed out waiting for the local Ollama chat API",
}

_SAMPLE_FILES_RESPONSE = [
    {
        "filename": "cli.py",
        "previous_filename": "cli.py",
        "patch": (
            '@@ -3,2 +3,2 @@\n'
            '-parser.add_argument("--timeout", help="Execution time")\n'
            '+parser.add_argument("--deadline", help="Execution time")\n'
        ),
    }
]


def _make_urlopen_mock(response_body: bytes, status: int = 200) -> MagicMock:
    """Return a mock context manager that yields a response with the given body."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen = MagicMock(return_value=mock_resp)
    return mock_urlopen


# ---------------------------------------------------------------------------
# Tests: _get_pr_diff
# ---------------------------------------------------------------------------


class TestGetPrDiff(unittest.TestCase):

    def test_successful_diff_assembly(self) -> None:
        body = json.dumps(_SAMPLE_FILES_RESPONSE).encode()
        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            _make_urlopen_mock(body),
        ):
            result = _get_pr_diff("owner/repo", 42, "token")

        self.assertIsNotNone(result)
        diff_text, truncated = result  # type: ignore[misc]
        self.assertFalse(truncated)
        self.assertIn("diff --git a/cli.py b/cli.py", diff_text)
        self.assertIn("--timeout", diff_text)
        self.assertIn("--deadline", diff_text)

    def test_404_returns_none(self) -> None:
        import urllib.error

        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.github.com/...",
                code=404,
                msg="Not Found",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            ),
        ):
            result = _get_pr_diff("owner/repo", 99, "token")

        self.assertIsNone(result)

    def test_malformed_json_returns_none(self) -> None:
        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            _make_urlopen_mock(b"not-json"),
        ):
            result = _get_pr_diff("owner/repo", 1, "token")

        self.assertIsNone(result)

    def test_unexpected_response_shape_returns_none(self) -> None:
        body = json.dumps({"not": "a list"}).encode()
        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            _make_urlopen_mock(body),
        ):
            result = _get_pr_diff("owner/repo", 1, "token")

        self.assertIsNone(result)

    def test_oversized_diff_is_truncated(self) -> None:
        """Files that together exceed MAX_DIFF_BYTES are truncated."""
        big_patch = "+" + "x" * (MAX_DIFF_BYTES + 1)
        files = [{"filename": "big.py", "previous_filename": "big.py", "patch": big_patch}]
        body = json.dumps(files).encode()
        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            _make_urlopen_mock(body),
        ):
            result = _get_pr_diff("owner/repo", 1, "token")

        # Should return something (the stub), not None, but marked truncated
        self.assertIsNotNone(result)
        diff_text, truncated = result  # type: ignore[misc]
        self.assertTrue(truncated)

    def test_empty_patch_fields_skipped(self) -> None:
        """Files without a 'patch' field (e.g. binaries) are silently skipped."""
        files = [{"filename": "image.png", "previous_filename": "image.png"}]
        body = json.dumps(files).encode()
        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            _make_urlopen_mock(body),
        ):
            result = _get_pr_diff("owner/repo", 1, "token")

        # Should return stub diff, not None
        self.assertIsNotNone(result)

    def test_connection_error_returns_none(self) -> None:
        import urllib.error

        with patch(
            "docdrift.github_reporter.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = _get_pr_diff("owner/repo", 1, "token")

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests: _format_report
# ---------------------------------------------------------------------------


class TestFormatReport(unittest.TestCase):

    def test_no_findings_shows_ok_message(self) -> None:
        report = _format_report(_NO_FINDINGS_RESULT, 7, use_ai=False)
        self.assertIn("No documentation drift detected", report)
        self.assertIn("PR #7", report) if "PR" in report else None

    def test_deterministic_finding_present(self) -> None:
        report = _format_report(_FLAG_RENAME_RESULT, 12, use_ai=False)
        self.assertIn("renamed_cli_flag", report)
        self.assertIn("--timeout", report)
        self.assertIn("--deadline", report)
        self.assertIn("README.md", report)
        self.assertIn("line 3", report)
        self.assertIn("Deterministic", report)

    def test_ai_suggestion_labeled_unverified(self) -> None:
        report = _format_report(_AI_RESULT, 5, use_ai=True)
        self.assertIn("Unverified Suggestions", report)
        self.assertIn("qwen3:4b", report)
        self.assertIn("unverified", report.lower())
        self.assertIn("possible_semantic_drift", report)
        self.assertIn("evidence_verified", report.lower()) if "evidence_verified" in report.lower() else self.assertIn("evidence verified", report.lower())

    def test_ai_verified_evidence_shows_verified_label(self) -> None:
        report = _format_report(_AI_RESULT, 5, use_ai=True)
        # evidence_verified=True in the fixture → should show verified indicator
        self.assertIn("Verified verbatim", report)

    def test_ai_unverified_evidence_shows_unverified_label(self) -> None:
        result_unverified = {
            **_AI_RESULT,
            "ai_analysis": {
                **_AI_RESULT["ai_analysis"],  # type: ignore[arg-type]
                "findings": [
                    {
                        **_AI_RESULT["ai_analysis"]["findings"][0],  # type: ignore[index]
                        "evidence_verified": False,
                    }
                ],
            },
        }
        report = _format_report(result_unverified, 5, use_ai=True)
        self.assertIn("Unverified", report)

    def test_ai_disabled_shows_note(self) -> None:
        report = _format_report(_NO_FINDINGS_RESULT, 1, use_ai=False)
        self.assertIn("AI analysis", report)
        self.assertIn("DOCDRIFT_AI_URL", report)

    def test_ai_failure_shown_gracefully(self) -> None:
        report = _format_report(_AI_FAILED_RESULT, 3, use_ai=True)
        self.assertIn("AI analysis failed", report)
        self.assertIn("Timed out", report)

    def test_diff_truncated_warning_appears(self) -> None:
        report = _format_report(_NO_FINDINGS_RESULT, 1, diff_truncated=True, use_ai=False)
        self.assertIn("truncated", report.lower())
        self.assertIn("512 KB", report)

    def test_documentation_files_footer(self) -> None:
        result = {**_AI_RESULT, "documentation_files": ["docs/config.md", "README.md"]}
        report = _format_report(result, 2, use_ai=True)
        self.assertIn("docs/config.md", report)
        self.assertIn("README.md", report)

    def test_report_contains_disclaimer(self) -> None:
        report = _format_report(_FLAG_RENAME_RESULT, 1, use_ai=False)
        self.assertIn("DocDrift", report)
        self.assertIn("human review", report)


# ---------------------------------------------------------------------------
# Tests: _post_pr_comment (deduplication)
# ---------------------------------------------------------------------------


class TestPostPrComment(unittest.TestCase):

    def test_existing_comment_deleted_before_posting(self) -> None:
        """When an existing DocDrift comment exists, it must be deleted first."""
        existing_comments = [
            {"id": 9001, "body": f"{COMMENT_MARKER}\n\nOld report text."},
            {"id": 9002, "body": "Some unrelated comment."},
        ]

        list_response_body = json.dumps(existing_comments).encode()
        delete_response_body = b""
        post_response_body = json.dumps({"id": 9003}).encode()

        call_count = {"n": 0}
        responses = [list_response_body, delete_response_body, post_response_body]

        def side_effect(req, timeout=30):
            mock_resp = MagicMock()
            mock_resp.read.return_value = responses[call_count["n"]]
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            call_count["n"] += 1
            return mock_resp

        with patch(
            "docdrift.github_reporter.urllib.request.urlopen", side_effect=side_effect
        ) as mock_urlopen:
            _post_pr_comment("owner/repo", 42, "token", "New report")

        # List + Delete + Post = 3 calls
        self.assertEqual(mock_urlopen.call_count, 3)
        # Third call should be a POST
        post_call_req = mock_urlopen.call_args_list[2][0][0]
        self.assertEqual(post_call_req.method, "POST")

    def test_no_previous_comment_only_posts(self) -> None:
        """When no previous DocDrift comment exists, only a POST is made."""
        existing_comments = [{"id": 1, "body": "Unrelated comment."}]
        responses = [
            json.dumps(existing_comments).encode(),
            json.dumps({"id": 2}).encode(),
        ]
        call_count = {"n": 0}

        def side_effect(req, timeout=30):
            mock_resp = MagicMock()
            mock_resp.read.return_value = responses[call_count["n"]]
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            call_count["n"] += 1
            return mock_resp

        with patch(
            "docdrift.github_reporter.urllib.request.urlopen", side_effect=side_effect
        ) as mock_urlopen:
            _post_pr_comment("owner/repo", 1, "token", "New report")

        # List + Post = 2 calls (no DELETE)
        self.assertEqual(mock_urlopen.call_count, 2)


# ---------------------------------------------------------------------------
# Tests: run_from_github_env (end-to-end, fully mocked)
# ---------------------------------------------------------------------------


class TestRunFromGithubEnv(unittest.TestCase):

    def _make_event_file(self, pr_number: int = 42) -> str:
        """Write a minimal PR event JSON to a temp file and return its path."""
        event = {
            "number": pr_number,
            "repository": {"full_name": "owner/testrepo"},
            "pull_request": {"number": pr_number},
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(event, tmp)
        tmp.flush()
        tmp.close()
        return tmp.name

    def _make_summary_file(self) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.close()
        return tmp.name

    def test_missing_github_token_returns_1(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            result = run_from_github_env()
        self.assertEqual(result, 1)

    def test_pr_with_cli_rename_writes_report(self) -> None:
        event_file = self._make_event_file(7)
        summary_file = self._make_summary_file()

        # GitHub API returns the sample file list
        api_body = json.dumps(_SAMPLE_FILES_RESPONSE).encode()

        with tempfile.TemporaryDirectory() as workspace:
            # Set up a minimal repo with a README that mentions the old flag
            (Path(workspace) / "README.md").write_text(
                "# CLI\n\nUse `--timeout` to stop slow jobs.\n", encoding="utf-8"
            )
            (Path(workspace) / "cli.py").write_text(
                'import argparse\nparser = argparse.ArgumentParser()\n'
                'parser.add_argument("--deadline", help="Execution time")\n',
                encoding="utf-8",
            )

            env = {
                "GITHUB_TOKEN": "ghs_fake_token",
                "GITHUB_REPOSITORY": "owner/testrepo",
                "GITHUB_EVENT_PATH": event_file,
                "GITHUB_WORKSPACE": workspace,
                "GITHUB_STEP_SUMMARY": summary_file,
            }

            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "docdrift.github_reporter.urllib.request.urlopen",
                    _make_urlopen_mock(api_body),
                ):
                    exit_code = run_from_github_env()

        with open(summary_file, encoding="utf-8") as fh:
            content = fh.read()

        self.assertEqual(exit_code, 0)
        self.assertIn("DocDrift", content)
        # Should detect the CLI rename
        self.assertIn("--timeout", content)

        os.unlink(event_file)
        os.unlink(summary_file)

    def test_no_findings_writes_ok_report(self) -> None:
        event_file = self._make_event_file(3)
        summary_file = self._make_summary_file()

        # Diff that touches a file with no documentation
        files = [{"filename": "utils.py", "patch": "@@ -1 +1 @@\n-x = 1\n+x = 2\n"}]
        api_body = json.dumps(files).encode()

        with tempfile.TemporaryDirectory() as workspace:
            (Path(workspace) / "README.md").write_text(
                "# Service\nNo flags documented here.\n", encoding="utf-8"
            )

            env = {
                "GITHUB_TOKEN": "ghs_fake",
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_EVENT_PATH": event_file,
                "GITHUB_WORKSPACE": workspace,
                "GITHUB_STEP_SUMMARY": summary_file,
            }

            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "docdrift.github_reporter.urllib.request.urlopen",
                    _make_urlopen_mock(api_body),
                ):
                    exit_code = run_from_github_env()

        with open(summary_file, encoding="utf-8") as fh:
            content = fh.read()

        self.assertEqual(exit_code, 0)
        self.assertIn("No documentation drift detected", content)

        os.unlink(event_file)
        os.unlink(summary_file)

    def test_api_failure_returns_1(self) -> None:
        import urllib.error

        event_file = self._make_event_file(5)

        with tempfile.TemporaryDirectory() as workspace:
            env = {
                "GITHUB_TOKEN": "ghs_fake",
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_EVENT_PATH": event_file,
                "GITHUB_WORKSPACE": workspace,
            }

            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "docdrift.github_reporter.urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "url", 500, "Server Error", None, None  # type: ignore[arg-type]
                    ),
                ):
                    exit_code = run_from_github_env()

        self.assertEqual(exit_code, 1)
        os.unlink(event_file)

    def test_malformed_event_json_returns_1(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("this is not json{{{{")
            event_file = tmp.name

        env = {
            "GITHUB_TOKEN": "ghs_fake",
            "GITHUB_EVENT_PATH": event_file,
        }

        with patch.dict(os.environ, env, clear=True):
            result = run_from_github_env()

        self.assertEqual(result, 1)
        os.unlink(event_file)

    def test_ai_disabled_by_default(self) -> None:
        """When DOCDRIFT_AI_URL is not set, analyze_changes is called with use_ai=False."""
        event_file = self._make_event_file(10)
        summary_file = self._make_summary_file()
        api_body = json.dumps([]).encode()

        with tempfile.TemporaryDirectory() as workspace:
            env = {
                "GITHUB_TOKEN": "ghs_fake",
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_EVENT_PATH": event_file,
                "GITHUB_WORKSPACE": workspace,
                "GITHUB_STEP_SUMMARY": summary_file,
            }
            # Ensure DOCDRIFT_AI_URL is absent
            env_clean = {k: v for k, v in env.items() if k != "DOCDRIFT_AI_URL"}

            with patch.dict(os.environ, env_clean, clear=True):
                with patch(
                    "docdrift.github_reporter.urllib.request.urlopen",
                    _make_urlopen_mock(api_body),
                ):
                    with patch(
                        "docdrift.github_reporter.analyze_changes",
                        return_value=_NO_FINDINGS_RESULT,
                    ) as mock_analyze:
                        run_from_github_env()

        call_kwargs = mock_analyze.call_args[1]
        self.assertFalse(call_kwargs.get("use_ai", True))

        os.unlink(event_file)
        os.unlink(summary_file)

    def test_ai_enabled_when_url_set(self) -> None:
        """When DOCDRIFT_AI_URL is set, analyze_changes is called with use_ai=True."""
        event_file = self._make_event_file(11)
        summary_file = self._make_summary_file()
        api_body = json.dumps([]).encode()

        with tempfile.TemporaryDirectory() as workspace:
            env = {
                "GITHUB_TOKEN": "ghs_fake",
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_EVENT_PATH": event_file,
                "GITHUB_WORKSPACE": workspace,
                "GITHUB_STEP_SUMMARY": summary_file,
                "DOCDRIFT_AI_URL": "http://myollama:11434/api/chat",
            }

            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "docdrift.github_reporter.urllib.request.urlopen",
                    _make_urlopen_mock(api_body),
                ):
                    with patch(
                        "docdrift.github_reporter.analyze_changes",
                        return_value=_NO_FINDINGS_RESULT,
                    ) as mock_analyze:
                        run_from_github_env()

        call_kwargs = mock_analyze.call_args[1]
        self.assertTrue(call_kwargs.get("use_ai"))
        self.assertEqual(
            call_kwargs.get("base_url"), "http://myollama:11434/api/chat"
        )

        os.unlink(event_file)
        os.unlink(summary_file)


# ---------------------------------------------------------------------------
# Tests: utilities
# ---------------------------------------------------------------------------


class TestUtilities(unittest.TestCase):

    def test_extract_pr_number_from_number_key(self) -> None:
        event = {"number": 42, "pull_request": {}}
        self.assertEqual(_extract_pr_number(event), 42)

    def test_extract_pr_number_from_pull_request_key(self) -> None:
        event = {"pull_request": {"number": 7}}
        self.assertEqual(_extract_pr_number(event), 7)

    def test_extract_pr_number_missing_returns_none(self) -> None:
        self.assertIsNone(_extract_pr_number({}))

    def test_load_github_event_valid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump({"number": 5}, tmp)
            event_file = tmp.name

        with patch.dict(os.environ, {"GITHUB_EVENT_PATH": event_file}):
            result = _load_github_event()

        self.assertEqual(result, {"number": 5})
        os.unlink(event_file)

    def test_load_github_event_invalid_json_returns_none(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("{bad json")
            event_file = tmp.name

        with patch.dict(os.environ, {"GITHUB_EVENT_PATH": event_file}):
            result = _load_github_event()

        self.assertIsNone(result)
        os.unlink(event_file)

    def test_load_github_event_missing_env_returns_empty_dict(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_EVENT_PATH"}
        with patch.dict(os.environ, env, clear=True):
            result = _load_github_event()
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Tests: Pagination and Link header handling
# ---------------------------------------------------------------------------


class TestPaginationAndLinkHeader(unittest.TestCase):

    def test_parse_next_link_valid(self) -> None:
        header = (
            '<https://api.github.com/repos/owner/repo/pulls/1/files?page=2&per_page=100>; rel="next", '
            '<https://api.github.com/repos/owner/repo/pulls/1/files?page=3&per_page=100>; rel="last"'
        )
        self.assertEqual(
            _parse_next_link(header),
            "https://api.github.com/repos/owner/repo/pulls/1/files?page=2&per_page=100",
        )

    def test_parse_next_link_no_next(self) -> None:
        header = '<https://api.github.com/repos/owner/repo/pulls/1/files?page=1&per_page=100>; rel="prev"'
        self.assertIsNone(_parse_next_link(header))

    def test_parse_next_link_empty_or_invalid(self) -> None:
        self.assertIsNone(_parse_next_link(""))
        self.assertIsNone(_parse_next_link(None))
        self.assertIsNone(_parse_next_link("invalid; rel=next"))

    def test_get_pr_diff_multi_page_pagination(self) -> None:
        """Verify that _get_pr_diff follows rel='next' across pages."""
        page1_files = [{"filename": "f1.py", "previous_filename": "f1.py", "patch": "@@ -1 +1 @@\n-a=1\n+a=2\n"}]
        page2_files = [{"filename": "f2.py", "previous_filename": "f2.py", "patch": "@@ -1 +1 @@\n-b=1\n+b=2\n"}]

        resp1 = MagicMock()
        resp1.read.return_value = json.dumps(page1_files).encode()
        resp1.headers = {"Link": '<https://api.github.com/next_page>; rel="next"'}
        resp1.__enter__ = lambda s: s
        resp1.__exit__ = MagicMock(return_value=False)

        resp2 = MagicMock()
        resp2.read.return_value = json.dumps(page2_files).encode()
        resp2.headers = {"Link": ""}
        resp2.__enter__ = lambda s: s
        resp2.__exit__ = MagicMock(return_value=False)

        pages = [resp1, resp2]
        call_idx = {"idx": 0}

        def fake_urlopen(req, timeout=30):
            res = pages[call_idx["idx"]]
            call_idx["idx"] += 1
            return res

        with patch("docdrift.github_reporter.urllib.request.urlopen", side_effect=fake_urlopen):
            diff_text, truncated = _get_pr_diff("owner/repo", 1, "token")  # type: ignore[misc]

        self.assertFalse(truncated)
        self.assertIn("f1.py", diff_text)
        self.assertIn("f2.py", diff_text)
        self.assertEqual(call_idx["idx"], 2)

    def test_post_pr_comment_multi_page_dedup(self) -> None:
        """Verify that comment cleanup paginates across multiple pages."""
        c1 = [{"id": 101, "body": "unrelated"}]
        c2 = [{"id": 102, "body": f"{COMMENT_MARKER}\nold report"}]

        resp_p1 = MagicMock()
        resp_p1.read.return_value = json.dumps(c1).encode()
        resp_p1.headers = {"Link": '<https://api.github.com/comments_page2>; rel="next"'}
        resp_p1.__enter__ = lambda s: s
        resp_p1.__exit__ = MagicMock(return_value=False)

        resp_p2 = MagicMock()
        resp_p2.read.return_value = json.dumps(c2).encode()
        resp_p2.headers = {"Link": ""}
        resp_p2.__enter__ = lambda s: s
        resp_p2.__exit__ = MagicMock(return_value=False)

        resp_del = MagicMock()
        resp_del.read.return_value = b""
        resp_del.__enter__ = lambda s: s
        resp_del.__exit__ = MagicMock(return_value=False)

        resp_post = MagicMock()
        resp_post.read.return_value = json.dumps({"id": 103}).encode()
        resp_post.__enter__ = lambda s: s
        resp_post.__exit__ = MagicMock(return_value=False)

        responses = [resp_p1, resp_p2, resp_del, resp_post]
        step = {"n": 0}

        def fake_urlopen(req, timeout=30):
            r = responses[step["n"]]
            step["n"] += 1
            return r

        with patch("docdrift.github_reporter.urllib.request.urlopen", side_effect=fake_urlopen) as mock_u:
            _post_pr_comment("owner/repo", 1, "token", "new report")

        self.assertEqual(step["n"], 4)
        del_req = mock_u.call_args_list[2][0][0]
        self.assertEqual(del_req.method, "DELETE")
        self.assertIn("102", del_req.full_url)


# ---------------------------------------------------------------------------
# Tests: Markdown sanitization and security
# ---------------------------------------------------------------------------


class TestSecurityAndSanitization(unittest.TestCase):

    def test_heading_injection_neutralized(self) -> None:
        injected = "# Heading 1\n## Heading 2\nNormal line"
        sanitized = _sanitize_markdown_text(injected)
        self.assertNotIn("\n# Heading 1", sanitized)
        self.assertIn("\\# Heading 1", sanitized)
        self.assertIn("\\## Heading 2", sanitized)
        self.assertIn("Normal line", sanitized)

    def test_table_pipe_injection_escaped(self) -> None:
        bad_val = "--option|break_table\nnew line"
        escaped = _escape_table_cell(bad_val)
        self.assertIn(r"\|break_table", escaped)
        self.assertNotIn("\n", escaped)

    def test_code_fence_escaped(self) -> None:
        injected_code = "print('hello')\n```\n# Injected Markdown\n```"
        fenced = _safe_code_fence(injected_code)
        self.assertEqual(fenced.count("```"), 2)  # only outer delimiters
        self.assertIn("'''", fenced)

    def test_runtime_api_base_override(self) -> None:
        with patch.dict(os.environ, {"DOCDRIFT_GITHUB_API": "https://ghe.mycompany.com/api/v3"}):
            self.assertEqual(_get_api_base(), "https://ghe.mycompany.com/api/v3")


if __name__ == "__main__":
    unittest.main()
