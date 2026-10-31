import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docdrift import SemanticDriftAnalysisError, analyze_changes, detect_drift
from docdrift.orchestrator import MAX_FILE_BYTES, MAX_TOTAL_DOC_BYTES


SAMPLE_DIFF_RENAME = """\
--- a/cli.py
+++ b/cli.py
@@ -10,3 +10,3 @@
-parser.add_argument("--old-flag", help="Enable feature")
+parser.add_argument("--new-flag", help="Enable feature")
"""

SAMPLE_DIFF_SEMANTIC = """\
--- a/config.py
+++ b/config.py
@@ -5,3 +5,3 @@
-DEFAULT_TIMEOUT = 30
+DEFAULT_TIMEOUT = 60
"""


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deterministic_only_mode(self):
        # Setup repo with a doc mentioning --old-flag
        doc = self.repo_dir / "README.md"
        doc.write_text("Run with `--old-flag` to enable.\n", encoding="utf-8")

        with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_RENAME, self.repo_dir, use_ai=False)
            mock_ai.assert_not_called()

        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["ai_analysis"])
        self.assertEqual(result["documentation_files"], [])
        self.assertIsNone(result["ai_error"])
        self.assertEqual(len(result["deterministic_findings"]), 1)
        self.assertEqual(result["deterministic_findings"][0]["old_flag"], "--old-flag")
        self.assertEqual(result["deterministic_findings"][0]["new_flag"], "--new-flag")

    def test_ai_mode_invokes_analyzer_with_relevant_doc(self):
        doc = self.repo_dir / "config.md"
        doc.write_text("The default_timeout is 30 seconds.\n", encoding="utf-8")

        mock_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": "The documented timeout changed from 30 to 60.",
                    "evidence": "Diff: `DEFAULT_TIMEOUT = 60` | Docs: `default_timeout is 30 seconds.`",
                    "suggestion": "Update documentation to 60 seconds.",
                    "confidence": "high",
                    "evidence_verified": True,
                }
            ],
            "limitations": ["Verified finding."],
            "model": "qwen3:4b",
            "status": "completed",
        }

        with patch("docdrift.orchestrator.analyze_semantic_drift", return_value=mock_payload) as mock_ai:
            result = analyze_changes(
                SAMPLE_DIFF_SEMANTIC,
                self.repo_dir,
                use_ai=True,
                model="custom:model",
                base_url="http://custom:11434",
                timeout=45.0,
            )
            mock_ai.assert_called_once()
            called_args, called_kwargs = mock_ai.call_args
            self.assertEqual(called_args[0], SAMPLE_DIFF_SEMANTIC)
            self.assertIn("default_timeout is 30 seconds", called_args[1])
            self.assertEqual(called_kwargs.get("model"), "custom:model")
            self.assertEqual(called_kwargs.get("base_url"), "http://custom:11434")
            self.assertEqual(called_kwargs.get("timeout"), 45.0)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["ai_analysis"], mock_payload)
        self.assertEqual(result["documentation_files"], ["config.md"])
        self.assertIsNone(result["ai_error"])

    def test_ai_mode_with_no_relevant_documentation(self):
        # Doc has no relevance to config.py or DEFAULT_TIMEOUT
        doc = self.repo_dir / "unrelated.md"
        doc.write_text("Bananas and apples are tropical fruits.\n", encoding="utf-8")

        with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
            mock_ai.assert_not_called()

        self.assertEqual(result["status"], "no_relevant_documentation")
        self.assertIsNone(result["ai_analysis"])
        self.assertEqual(result["documentation_files"], [])
        self.assertIsNone(result["ai_error"])
        self.assertEqual(result["deterministic_findings"], [])

    def test_ai_analyzer_failure_preserves_deterministic_results(self):
        doc = self.repo_dir / "README.md"
        doc.write_text("Use `--old-flag` for configuration.\n", encoding="utf-8")

        with patch(
            "docdrift.orchestrator.analyze_semantic_drift",
            side_effect=SemanticDriftAnalysisError("Connection refused to Ollama"),
        ) as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_RENAME, self.repo_dir, use_ai=True)
            mock_ai.assert_called_once()

        self.assertEqual(result["status"], "ai_failed")
        self.assertIsNone(result["ai_analysis"])
        self.assertEqual(result["documentation_files"], ["README.md"])
        self.assertIn("Connection refused", result["ai_error"])
        # Deterministic findings are strictly preserved
        self.assertEqual(len(result["deterministic_findings"]), 1)
        self.assertEqual(result["deterministic_findings"][0]["old_flag"], "--old-flag")

    def test_invalid_inputs_raise(self):
        # Invalid diff inputs
        for bad_diff in [None, 123, [], {}, True]:
            with self.assertRaises(ValueError):
                analyze_changes(bad_diff, self.repo_dir)

        # Invalid repo_path inputs
        for bad_repo in [None, 123, Path("/nonexistent/repo/dir/12345"), "/nonexistent/repo/dir/12345"]:
            with self.assertRaises(ValueError):
                analyze_changes(SAMPLE_DIFF_RENAME, bad_repo)

        # Invalid timeout inputs
        for bad_timeout in [0, -1, -50.5, "invalid", None, False, True]:
            with self.assertRaises(ValueError):
                analyze_changes(SAMPLE_DIFF_RENAME, self.repo_dir, timeout=bad_timeout)

        # Invalid use_ai inputs
        for bad_use_ai in ["yes", "false", 1, 0, None, [], {}]:
            with self.assertRaises(ValueError):
                analyze_changes(SAMPLE_DIFF_RENAME, self.repo_dir, use_ai=bad_use_ai)

    def test_empty_diff_raises(self):
        for empty_diff in ["", "   ", "\t\n  \n"]:
            with self.assertRaises(ValueError):
                analyze_changes(empty_diff, self.repo_dir)

    def test_missing_or_unreadable_doc_files_handled_gracefully(self):
        doc = self.repo_dir / "config.md"
        doc.write_text("The default_timeout is 30 seconds.\n", encoding="utf-8")

        # Mock open to raise PermissionError when opening config.md
        original_open = open

        def faulty_open(path, *args, **kwargs):
            if "config.md" in str(path):
                raise PermissionError("Access denied")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=faulty_open):
            with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
                result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
                mock_ai.assert_not_called()

        self.assertEqual(result["status"], "no_relevant_documentation")
        self.assertEqual(result["documentation_files"], [])
        self.assertIsNone(result["ai_analysis"])

    def test_large_documentation_files_handled_safely(self):
        doc = self.repo_dir / "config.md"
        # Create a file larger than MAX_FILE_BYTES (64KB)
        padding = "default_timeout is 30 seconds.\n" + ("x" * 70000)
        doc.write_text(padding, encoding="utf-8")

        with patch("docdrift.orchestrator.analyze_semantic_drift", return_value={"findings": []}) as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
            mock_ai.assert_called_once()
            called_docs = mock_ai.call_args[0][1]
            self.assertIn("... [content truncated due to size limit] ...", called_docs)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["documentation_files"], ["config.md"])

    def test_backward_compatibility_detect_drift(self):
        # Directly call detect_drift to ensure original signature and contract are untouched
        doc = self.repo_dir / "cli.md"
        doc.write_text("Use `--old-flag` option.\n", encoding="utf-8")

        findings = detect_drift(SAMPLE_DIFF_RENAME, self.repo_dir)
        self.assertIsInstance(findings, list)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["old_flag"], "--old-flag")
        self.assertEqual(finding["new_flag"], "--new-flag")
        self.assertEqual(finding["documentation_file"], "cli.md")
        self.assertEqual(finding["line_number"], 1)

    def test_relevance_ranking_selects_most_relevant_first(self):
        # File 1: Exact stem match and identifier mention
        doc1 = self.repo_dir / "config.md"
        doc1.write_text("DEFAULT_TIMEOUT configuration details.\n", encoding="utf-8")

        # File 2: Identifier match only
        doc2 = self.repo_dir / "guide.md"
        doc2.write_text("DEFAULT_TIMEOUT is mentioned here.\n", encoding="utf-8")

        with patch("docdrift.orchestrator.analyze_semantic_drift", return_value={"findings": []}) as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
            mock_ai.assert_called_once()

        self.assertIn("config.md", result["documentation_files"])
        # config.md should come first because of stem match score bonus (+40)
        self.assertEqual(result["documentation_files"][0], "config.md")

    def test_symlinks_outside_repo_ignored(self):
        with tempfile.TemporaryDirectory() as outside_td:
            outside_doc = Path(outside_td) / "outside.md"
            outside_doc.write_text("DEFAULT_TIMEOUT is 30.\n", encoding="utf-8")
            sym = self.repo_dir / "symlink_outside.md"
            sym.symlink_to(outside_doc)

            with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
                result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
                mock_ai.assert_not_called()

            self.assertEqual(result["status"], "no_relevant_documentation")
            self.assertEqual(result["documentation_files"], [])

    def test_symlinks_to_hidden_files_ignored(self):
        hidden_dir = self.repo_dir / ".hidden"
        hidden_dir.mkdir()
        hidden_doc = hidden_dir / "secret.md"
        hidden_doc.write_text("DEFAULT_TIMEOUT is 30.\n", encoding="utf-8")
        sym = self.repo_dir / "symlink_hidden.md"
        sym.symlink_to(hidden_doc)

        with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
            result = analyze_changes(SAMPLE_DIFF_SEMANTIC, self.repo_dir, use_ai=True)
            mock_ai.assert_not_called()

        self.assertEqual(result["status"], "no_relevant_documentation")
        self.assertEqual(result["documentation_files"], [])

    def test_diff_headers_do_not_falsely_match_diff_md(self):
        doc = self.repo_dir / "diff.md"
        doc.write_text("This documentation discusses unrelated topics.\n", encoding="utf-8")

        diff = """\
diff --git a/worker.py b/worker.py
--- a/worker.py
+++ b/worker.py
@@ -1,2 +1,2 @@
-foo = 1
+foo = 2
"""
        with patch("docdrift.orchestrator.analyze_semantic_drift") as mock_ai:
            result = analyze_changes(diff, self.repo_dir, use_ai=True)
            mock_ai.assert_not_called()

        self.assertEqual(result["status"], "no_relevant_documentation")
        self.assertEqual(result["documentation_files"], [])


if __name__ == "__main__":
    unittest.main()
