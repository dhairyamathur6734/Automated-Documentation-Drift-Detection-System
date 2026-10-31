import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docdrift.detector import detect_drift

SAMPLE_DIFF = """\
diff --git a/cli.py b/cli.py
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""

OUTPUT_DIFF = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--output", help="Write results")
+parser.add_argument("--destination", help="Write results")
"""

SHORT_FLAG_DIFF = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("-t", help="Maximum execution time")
+parser.add_argument("-d", help="Maximum execution time")
"""


def _write_markdown(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class DetectDriftTests(unittest.TestCase):
    def test_sample_diff_detects_outdated_timeout_reference(self) -> None:
        repo = ROOT / "examples" / "sample_repo"
        findings = detect_drift(SAMPLE_DIFF, repo)

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["type"], "renamed_cli_flag")
        self.assertEqual(finding["old_flag"], "--timeout")
        self.assertEqual(finding["new_flag"], "--deadline")
        self.assertEqual(finding["documentation_file"], "README.md")
        self.assertEqual(finding["line_number"], 3)
        self.assertEqual(finding["outdated_text"], "Use `--timeout` to stop slow jobs.")
        self.assertEqual(
            finding["code_change"]["removed"],
            'parser.add_argument("--timeout", help="Maximum execution time")',
        )
        self.assertEqual(
            finding["code_change"]["added"],
            'parser.add_argument("--deadline", help="Maximum execution time")',
        )

    def test_correct_deadline_reference_is_not_reported(self) -> None:
        repo = ROOT / "examples" / "sample_repo"
        findings = detect_drift(SAMPLE_DIFF, repo)
        texts = [item["outdated_text"] for item in findings]
        self.assertFalse(
            any("--deadline" in text and "--timeout" not in text for text in texts)
        )
        self.assertEqual(
            [item["line_number"] for item in findings],
            [3],
        )

    def test_multiple_outdated_occurrences_are_reported_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(
                root,
                "README.md",
                "Use `--timeout` first.\n"
                "Then use `--timeout` again.\n"
                "Both `--timeout` mentions share one line.\n",
            )
            findings = detect_drift(SAMPLE_DIFF, root)

        self.assertEqual([item["line_number"] for item in findings], [1, 2, 3])
        self.assertEqual({item["old_flag"] for item in findings}, {"--timeout"})

    def test_nested_markdown_files_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "docs/usage.md", "Set `--timeout` in docs.\n")
            _write_markdown(root, "examples/README.md", "Example uses `--timeout`.\n")
            findings = detect_drift(SAMPLE_DIFF, root)

        paths = [item["documentation_file"] for item in findings]
        self.assertEqual(paths, ["docs/usage.md", "examples/README.md"])
        self.assertEqual([item["line_number"] for item in findings], [1, 1])

    def test_multiple_independent_unambiguous_renames(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,6 +1,6 @@
 import argparse
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
 parser.add_argument("--verbose")
-parser.add_argument("--output", help="Write results")
+parser.add_argument("--destination", help="Write results")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(
                root,
                "README.md",
                "Use `--timeout` and `--output`.\nKeep `--verbose`.\n",
            )
            findings = detect_drift(diff, root)

        pairs = [(item["old_flag"], item["new_flag"], item["line_number"]) for item in findings]
        self.assertEqual(
            pairs,
            [("--output", "--destination", 1), ("--timeout", "--deadline", 1)],
        )

    def test_different_flag_rename_without_hardcoded_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Write to `--output` or `--destination`.\n")
            findings = detect_drift(OUTPUT_DIFF, root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["old_flag"], "--output")
        self.assertEqual(findings[0]["new_flag"], "--destination")
        self.assertIn("`--output`", findings[0]["outdated_text"])
        self.assertNotIn("--output", json.dumps(findings[0]["code_change"]["added"]))

    def test_short_flag_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Pass `-t` to stop early. `-d` is current.\n")
            findings = detect_drift(SHORT_FLAG_DIFF, root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["old_flag"], "-t")
        self.assertEqual(findings[0]["new_flag"], "-d")
        self.assertEqual(findings[0]["line_number"], 1)

    def test_empty_diff_returns_no_findings(self) -> None:
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift("", repo), [])
        self.assertEqual(detect_drift("   \n", repo), [])

    def test_diff_without_recognizable_rename_returns_no_findings(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,4 @@
 import argparse
 parser.add_argument("--deadline", help="Maximum execution time")
+print("hello")
"""
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift(diff, repo), [])

    def test_documentation_with_only_new_flag_returns_no_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--deadline` only.\n")
            self.assertEqual(detect_drift(SAMPLE_DIFF, root), [])

    def test_documentation_without_old_flag_returns_no_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "No CLI flags are documented here.\n")
            self.assertEqual(detect_drift(SAMPLE_DIFF, root), [])

    def test_old_flag_embedded_in_longer_option_is_not_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "See `--timeout-extra` and `--timeout2`.\n")
            self.assertEqual(detect_drift(SAMPLE_DIFF, root), [])

    def test_hyphenated_words_are_not_mistaken_for_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "This is well-known time-out behaviour.\n")
            self.assertEqual(detect_drift(SAMPLE_DIFF, root), [])

    def test_missing_repository_path_is_handled_gracefully(self) -> None:
        missing = Path("/tmp/docdrift-missing-repo-path-does-not-exist")
        self.assertEqual(detect_drift(SAMPLE_DIFF, missing), [])

    def test_empty_repository_returns_no_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(detect_drift(SAMPLE_DIFF, Path(tmp)), [])

    def test_malformed_diffs_do_not_crash(self) -> None:
        repo = ROOT / "examples" / "sample_repo"
        malformed = [
            "not a diff at all",
            "@@ incomplete hunk",
            "+++ b/cli.py\n-parser.add_argument('--timeout')",
            "diff --git a/cli.py b/cli.py\nrandom metadata\n",
        ]
        for diff in malformed:
            with self.subTest(diff=diff):
                self.assertEqual(detect_drift(diff, repo), [])

    def test_ambiguous_rename_relationships_are_not_invented(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,4 +1,4 @@
 import argparse
-parser.add_argument("--timeout", help="desc")
-parser.add_argument("--output", help="desc")
+parser.add_argument("--deadline", help="desc")
+parser.add_argument("--destination", help="desc")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--timeout` and `--output`.\n")
            self.assertEqual(detect_drift(diff, root), [])

    def test_repeated_runs_are_deterministic(self) -> None:
        repo = ROOT / "examples" / "sample_repo"
        first = detect_drift(SAMPLE_DIFF, repo)
        second = detect_drift(SAMPLE_DIFF, repo)
        self.assertEqual(first, second)

    def test_quoted_flag_literal_continuation_is_paired(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,5 +1,5 @@
 parser.add_argument(
-    "--old-name",
+    "--new-name",
     help="label",
 )
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Pass `--old-name` here.\n")
            findings = detect_drift(diff, root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["old_flag"], "--old-name")
        self.assertEqual(findings[0]["new_flag"], "--new-name")

    def test_competing_declaration_candidates_are_not_greedily_paired(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,4 +1,3 @@
 import argparse
-parser.add_argument("--timeout", help="Maximum execution time")
-parser.add_argument("--retries", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--timeout` or `--retries`.\n")
            self.assertEqual(detect_drift(diff, root), [])

    def test_reordered_aliases_are_not_paired_by_position(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", "-t")
+parser.add_argument("-d", "--deadline")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--timeout` or `-t`.\n")
            findings = detect_drift(diff, root)

        pairs = {(item["old_flag"], item["new_flag"]) for item in findings}
        self.assertNotIn(("--timeout", "-d"), pairs)
        self.assertNotIn(("-t", "--deadline"), pairs)
        self.assertEqual(pairs, {("--timeout", "--deadline"), ("-t", "-d")})

    def test_two_long_flags_on_one_declaration_are_not_zipped(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", "--max-time")
+parser.add_argument("--deadline", "--until")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--timeout` or `--max-time`.\n")
            self.assertEqual(detect_drift(diff, root), [])

    def test_headerless_minus_plus_lines_do_not_create_findings(self) -> None:
        diff = """\
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift(diff, repo), [])

    def test_changes_outside_a_valid_hunk_do_not_create_findings(self) -> None:
        diff = """\
diff --git a/cli.py b/cli.py
--- a/cli.py
+++ b/cli.py
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift(diff, repo), [])

    def test_invalid_hunk_header_does_not_create_findings(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ incomplete hunk
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift(diff, repo), [])

    def test_documentation_file_diff_does_not_infer_rename(self) -> None:
        diff = """\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,3 +1,3 @@
 # CLI
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
"""
        repo = ROOT / "examples" / "sample_repo"
        self.assertEqual(detect_drift(diff, repo), [])

    def test_conflicting_old_flag_mappings_are_skipped(self) -> None:
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", help="Maximum execution time")
+parser.add_argument("--deadline", help="Maximum execution time")
--- a/worker.py
+++ b/worker.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", help="Worker budget")
+parser.add_argument("--expiry", help="Worker budget")
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", "Use `--timeout` before the job ends.\n")
            self.assertEqual(detect_drift(diff, root), [])

    def test_false_positive_diff_additions_and_schemas_ignored(self) -> None:
        """Diff addition code lines and finding schemas must not be flagged."""
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--deadline", help="Maximum execution time")
+parser.add_argument("--timeout-limit", help="Maximum execution time")
"""
        doc = (
            "# Code Examples and Schemas\n\n"
            "+parser.add_argument(\"--deadline\", help=\"Maximum execution time\")\n"
            "+parser.add_argument(\"--deadline\", default=60)\n"
            "    \"new_flag\": \"--deadline\",\n"
            "    \"added\": 'parser.add_argument(\"--deadline\")',\n"
            "Run the program with `--deadline` to set the maximum execution time.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", doc)
            findings = detect_drift(diff, root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line_number"], 7)
        self.assertEqual(
            findings[0]["outdated_text"],
            "Run the program with `--deadline` to set the maximum execution time.",
        )

    def test_false_positive_different_rename_and_correct_assertions_ignored(self) -> None:
        """Historical explanations of other renames and explicit correctness assertions must not be flagged."""
        diff = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--deadline", help="Maximum execution time")
+parser.add_argument("--timeout-limit", help="Maximum execution time")
"""
        doc = (
            "- Code changes `--timeout` to `--deadline`.\n"
            "A unified diff renaming `--timeout` to `--deadline` should produce one finding.\n"
            "The sample README also contains a correct `--deadline` reference. That line is not reported.\n"
            "The `--deadline` sentence must not be reported.\n"
            "contains code that defines `--deadline` while also correctly mentioning `--deadline`.\n"
            "Use `--deadline` in production.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_markdown(root, "README.md", doc)
            findings = detect_drift(diff, root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line_number"], 6)
        self.assertEqual(findings[0]["outdated_text"], "Use `--deadline` in production.")


if __name__ == "__main__":
    unittest.main()
