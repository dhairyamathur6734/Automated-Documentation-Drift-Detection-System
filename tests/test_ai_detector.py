import io
import json
import socket
import sys
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docdrift.ai_detector import (
    DEFAULT_MODEL,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT_SECONDS,
    SemanticDriftAnalysisError,
    analyze_semantic_drift,
)

SAMPLE_DIFF = """\
--- a/cli.py
+++ b/cli.py
@@ -1,3 +1,3 @@
 import argparse
-parser.add_argument("--timeout", default=30)
+parser.add_argument("--timeout", default=60)
"""
SAMPLE_DOCS = "The timeout defaults to 30 seconds.\n"
VALID_MODEL_PAYLOAD = {
    "findings": [
        {
            "type": "possible_semantic_drift",
            "description": "The documented timeout may be outdated.",
            "evidence": 'Diff: `parser.add_argument("--timeout", default=60)` | Docs: `The timeout defaults to 30 seconds.`',
            "suggestion": "Verify the documented timeout and correct it if necessary.",
            "confidence": "medium",
            "evidence_verified": True,
        }
    ],
    "limitations": ["This is an unverified model suggestion."],
}


def _urlopen_response(envelope: dict) -> MagicMock:
    body = json.dumps(envelope).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _envelope(content: dict | str) -> dict:
    if isinstance(content, dict):
        content = json.dumps(content)
    return {"message": {"role": "assistant", "content": content}}


class AnalyzeSemanticDriftTests(unittest.TestCase):
    def test_empty_diff_raises_before_network_access(self) -> None:
        with patch("docdrift.ai_detector.urllib.request.urlopen") as mock_urlopen:
            with self.assertRaises(ValueError):
                analyze_semantic_drift("", SAMPLE_DOCS)
            with self.assertRaises(ValueError):
                analyze_semantic_drift("   \n", SAMPLE_DOCS)
        mock_urlopen.assert_not_called()

    def test_empty_documentation_raises_before_network_access(self) -> None:
        with patch("docdrift.ai_detector.urllib.request.urlopen") as mock_urlopen:
            with self.assertRaises(ValueError):
                analyze_semantic_drift(SAMPLE_DIFF, "")
            with self.assertRaises(ValueError):
                analyze_semantic_drift(SAMPLE_DIFF, " \t")
        mock_urlopen.assert_not_called()

    def test_non_string_diff_raises_before_network_access(self) -> None:
        with patch("docdrift.ai_detector.urllib.request.urlopen") as mock_urlopen:
            with self.assertRaises(ValueError):
                analyze_semantic_drift(None, SAMPLE_DOCS)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                analyze_semantic_drift(123, SAMPLE_DOCS)  # type: ignore[arg-type]
        mock_urlopen.assert_not_called()

    def test_non_string_documentation_raises_before_network_access(self) -> None:
        with patch("docdrift.ai_detector.urllib.request.urlopen") as mock_urlopen:
            with self.assertRaises(ValueError):
                analyze_semantic_drift(SAMPLE_DIFF, None)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                analyze_semantic_drift(SAMPLE_DIFF, 123)  # type: ignore[arg-type]
        mock_urlopen.assert_not_called()

    def test_successful_mocked_response_preserves_findings(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(VALID_MODEL_PAYLOAD)),
        ) as mock_urlopen:
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        mock_urlopen.assert_called_once()
        self.assertEqual(result["status"], "unverified_ai_suggestions")
        self.assertEqual(result["model"], DEFAULT_MODEL)
        self.assertEqual(result["findings"], VALID_MODEL_PAYLOAD["findings"])
        self.assertEqual(result["limitations"], VALID_MODEL_PAYLOAD["limitations"])
        json.dumps(result)

    def test_empty_findings_are_valid(self) -> None:
        payload = {"findings": [], "limitations": ["No contradiction found."]}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(result["findings"], [])
        self.assertEqual(result["limitations"], ["No contradiction found."])
        self.assertEqual(result["status"], "unverified_ai_suggestions")

    def test_default_model_and_url_are_used(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
        ) as mock_urlopen:
            analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        request = mock_urlopen.call_args[0][0]
        self.assertIsInstance(request, urllib.request.Request)
        self.assertEqual(request.full_url, DEFAULT_OLLAMA_URL)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertIn("options", payload)
        self.assertEqual(payload["options"]["num_predict"], DEFAULT_NUM_PREDICT)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn(SAMPLE_DIFF, payload["messages"][1]["content"])
        self.assertIn(SAMPLE_DOCS, payload["messages"][1]["content"])
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 60.0)

    def test_configurable_timeout_explicit_and_env(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
        ) as mock_urlopen:
            analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS, timeout=12.5)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 12.5)

        with patch.dict("os.environ", {"DOCDRIFT_TIMEOUT": "25.0"}, clear=False):
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
            ) as mock_urlopen:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
            self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 25.0)

            # Explicit argument overrides env var
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
            ) as mock_urlopen:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS, timeout=5.0)
            self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 5.0)

    def test_num_predict_configurable_via_env(self) -> None:
        with patch.dict("os.environ", {"DOCDRIFT_AI_NUM_PREDICT": "128"}, clear=False):
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
            ) as mock_urlopen:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        request = mock_urlopen.call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["options"]["num_predict"], 128)

    def test_explicit_parameters_override_environment(self) -> None:
        env = {
            "DOCDRIFT_OLLAMA_URL": "http://env.example/api/chat",
            "DOCDRIFT_AI_MODEL": "env-model",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(
                    _envelope({"findings": [], "limitations": []})
                ),
            ) as mock_urlopen:
                result = analyze_semantic_drift(
                    SAMPLE_DIFF,
                    SAMPLE_DOCS,
                    model="explicit-model",
                    base_url="http://explicit.local/api/chat",
                )

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://explicit.local/api/chat")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "explicit-model")
        self.assertEqual(result["model"], "explicit-model")

    def test_positional_parameters_override_environment(self) -> None:
        env = {
            "DOCDRIFT_OLLAMA_URL": "http://env.example/api/chat",
            "DOCDRIFT_AI_MODEL": "env-model",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(
                    _envelope({"findings": [], "limitations": []})
                ),
            ) as mock_urlopen:
                result = analyze_semantic_drift(
                    SAMPLE_DIFF,
                    SAMPLE_DOCS,
                    "explicit-model",
                    "http://explicit.local/api/chat",
                )

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://explicit.local/api/chat")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "explicit-model")
        self.assertEqual(result["model"], "explicit-model")

    def test_base_url_without_endpoint_is_normalized(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(
                _envelope({"findings": [], "limitations": []})
            ),
        ) as mock_urlopen:
            analyze_semantic_drift(
                SAMPLE_DIFF,
                SAMPLE_DOCS,
                base_url="http://localhost:11434",
            )

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")

    def test_environment_variables_override_defaults(self) -> None:
        env = {
            "DOCDRIFT_OLLAMA_URL": "http://env.example/api/chat",
            "DOCDRIFT_AI_MODEL": "env-model",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch(
                "docdrift.ai_detector.urllib.request.urlopen",
                return_value=_urlopen_response(
                    _envelope({"findings": [], "limitations": []})
                ),
            ) as mock_urlopen:
                result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://env.example/api/chat")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "env-model")
        self.assertEqual(result["model"], "env-model")

    def test_connection_failure_raises(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("connect", str(ctx.exception).lower())
        self.assertNotIn("SECRET", str(ctx.exception))

    def test_timeout_raises(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            side_effect=TimeoutError(),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("timed out", str(ctx.exception).lower())

        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            side_effect=urllib.error.URLError(socket.timeout("timed out")),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_http_error_raises(self) -> None:
        error = urllib.error.HTTPError(
            DEFAULT_OLLAMA_URL,
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with patch("docdrift.ai_detector.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("500", str(ctx.exception))

    def test_invalid_response_json_raises(self) -> None:
        response = MagicMock()
        response.read.return_value = b"not-json"
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("docdrift.ai_detector.urllib.request.urlopen", return_value=response):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("valid JSON", str(ctx.exception))

    def test_missing_message_content_raises(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response({"message": {"role": "assistant"}}),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("message.content", str(ctx.exception))

    def test_malformed_model_json_raises(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope("this is not json")),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("Model output", str(ctx.exception))

    def test_invalid_findings_type_raises(self) -> None:
        payload = {"findings": "not-a-list", "limitations": []}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_invalid_limitations_type_raises(self) -> None:
        payload = {"findings": [], "limitations": "nope"}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_invalid_finding_field_types_raise(self) -> None:
        payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": 123,
                    "evidence": "x",
                    "suggestion": "y",
                    "confidence": "low",
                }
            ],
            "limitations": [],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_missing_finding_fields_raise(self) -> None:
        payload = {
            "findings": [{"type": "possible_semantic_drift", "description": "x"}],
            "limitations": [],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_missing_top_level_fields_raise(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope({"findings": []})),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_empty_message_content_raises(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response({"message": {"role": "assistant", "content": "   "}}),
        ):
            with self.assertRaises(SemanticDriftAnalysisError) as ctx:
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)
        self.assertIn("non-empty string", str(ctx.exception))

    def test_fenced_json_parsing(self) -> None:
        fenced_payload = (
            "Here is the analysis:\n"
            "```json\n"
            + json.dumps(VALID_MODEL_PAYLOAD)
            + "\n```\nHope this helps!"
        )
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(fenced_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(result["findings"], VALID_MODEL_PAYLOAD["findings"])
        self.assertEqual(result["limitations"], VALID_MODEL_PAYLOAD["limitations"])

    def test_think_tags_parsing(self) -> None:
        think_payload = (
            "<think>Analyzing code diff vs docs carefully...</think>\n"
            + json.dumps(VALID_MODEL_PAYLOAD)
        )
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(think_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(result["findings"], VALID_MODEL_PAYLOAD["findings"])

    def test_invalid_finding_element_not_dict_raises(self) -> None:
        payload = {"findings": ["not-a-dict"], "limitations": []}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_invalid_limitation_element_not_string_raises(self) -> None:
        payload = {"findings": [], "limitations": [123]}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_empty_finding_field_raises(self) -> None:
        payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": "   ",
                    "evidence": "x",
                    "suggestion": "y",
                    "confidence": "low",
                }
            ],
            "limitations": [],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(payload)),
        ):
            with self.assertRaises(SemanticDriftAnalysisError):
                analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

    def test_prompt_injection_defense_instructions_present(self) -> None:
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
        ) as mock_urlopen:
            analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        request = mock_urlopen.call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        system_content = payload["messages"][0]["content"]
        user_content = payload["messages"][1]["content"]

        self.assertIn("untrusted", system_content.lower())
        self.assertIn("untrusted", user_content.lower())

    def test_fabricated_evidence_is_marked_unverified(self) -> None:
        # Replicate real-world qwen3:4b hallucinated quote: "The default timeout is 3: 30 seconds."
        fabricated_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": "Timeout inconsistency detected.",
                    "evidence": "The default timeout is 3: 30 seconds.",
                    "suggestion": "Update documentation to reflect the new default.",
                    "confidence": "high",
                }
            ],
            "limitations": ["Initial limitation."],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(fabricated_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        self.assertFalse(finding["evidence_verified"])
        self.assertEqual(finding["evidence"], "The default timeout is 3: 30 seconds.")
        # Limitations should explicitly record that evidence was unverified
        self.assertTrue(
            any("unverified evidence" in lim.lower() for lim in result["limitations"])
        )

    def test_verbatim_quoted_evidence_is_marked_verified(self) -> None:
        verbatim_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": "Timeout inconsistency detected.",
                    "evidence": "The timeout defaults to 30 seconds.",
                    "suggestion": "Update documentation to 60 seconds.",
                    "confidence": "high",
                }
            ],
            "limitations": ["No model limitations."],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(verbatim_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        self.assertTrue(finding["evidence_verified"])
        self.assertEqual(result["limitations"], ["No model limitations."])

    def test_mixed_findings_evidence_verification(self) -> None:
        mixed_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": "Valid finding with exact quote.",
                    "evidence": '`parser.add_argument("--timeout", default=60)`',
                    "suggestion": "Keep updated.",
                    "confidence": "high",
                },
                {
                    "type": "possible_semantic_drift",
                    "description": "Finding with fabricated quote.",
                    "evidence": 'Docs state "The timeout is 9999 seconds."',
                    "suggestion": "Fix timeout.",
                    "confidence": "low",
                },
            ],
            "limitations": [],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(mixed_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(len(result["findings"]), 2)
        self.assertTrue(result["findings"][0]["evidence_verified"])
        self.assertFalse(result["findings"][1]["evidence_verified"])
        self.assertTrue(
            any("unverified evidence" in lim.lower() for lim in result["limitations"])
        )

    # ---------------------------------------------------------------------------
    # Regression tests: prompt behaviour for behavior-preserving refactorings
    # ---------------------------------------------------------------------------

    def test_empty_findings_accepted(self) -> None:
        """A mocked model response with findings=[] passes validation without error."""
        empty_payload = {"findings": [], "limitations": ["No drift detected."]}
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(empty_payload)),
        ):
            result = analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        self.assertEqual(result["findings"], [])
        self.assertEqual(result["status"], "unverified_ai_suggestions")

    def test_prompt_contains_refactoring_guidance(self) -> None:
        """System and user prompts must instruct the model not to flag refactorings."""
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope({"findings": [], "limitations": []})),
        ) as mock_urlopen:
            analyze_semantic_drift(SAMPLE_DIFF, SAMPLE_DOCS)

        request = mock_urlopen.call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        system_content = payload["messages"][0]["content"]
        user_content = payload["messages"][1]["content"]

        # Security notice still present
        self.assertIn("untrusted", system_content.lower())
        self.assertIn("untrusted", user_content.lower())

        # New refactoring guidance present in system prompt
        self.assertIn("REFACTORING IS NOT DRIFT", system_content)
        self.assertIn("RIGOROUS EDGE-CASE TRACING", system_content)
        self.assertIn("EXPLICIT CONTRADICTIONS ONLY", system_content)

        # User prompt reminds model to return empty list for refactorings
        self.assertIn("behavior-preserving refactoring", user_content)

    def test_behavior_preserving_refactor_empty_findings_accepted(self) -> None:
        """
        When the model correctly returns empty findings for a behavior-preserving
        refactor, the orchestrator must not add spurious findings or raise.
        This uses a mocked response matching what the live model produced with
        the improved prompt for the loop-to-comprehension scenario.
        """
        refactor_diff = """\
diff --git a/service.py b/service.py
--- a/service.py
+++ b/service.py
@@ -1,5 +1,2 @@
 def process_batch(items):
-    res = []
-    for item in items:
-        res.append(item.strip())
-    return res
+    return [item.strip() for item in items]
"""
        refactor_docs = (
            "# Batch Processor\n\n"
            "Call `process_batch(items)` to sanitize a list of string elements.\n"
            "Returns processed items with whitespace removed.\n"
        )
        # Simulates what the improved-prompt model should return: no drift
        empty_payload = {
            "findings": [],
            "limitations": ["Refactoring preserves existing documented behavior; no drift detected."],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(empty_payload)),
        ):
            result = analyze_semantic_drift(refactor_diff, refactor_docs)

        self.assertEqual(result["findings"], [])
        self.assertIsNotNone(result["limitations"])

    def test_genuine_default_value_drift_finding_accepted(self) -> None:
        """
        When the model correctly reports a default value change (30->60) that
        contradicts the documentation, the finding must be returned intact.
        """
        default_diff = """\
diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1,2 +1,2 @@
-DEFAULT_TIMEOUT = 30
+DEFAULT_TIMEOUT = 60
"""
        default_docs = (
            "# Configuration Guide\n\n"
            "The connection timeout defaults to 30 seconds.\n"
            "Retry attempts default to 3.\n"
        )
        genuine_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": (
                        "The documentation states that the connection timeout defaults to 30 seconds, "
                        "but the code change increased the default timeout to 60 seconds."
                    ),
                    "evidence": "DEFAULT_TIMEOUT = 30\n+DEFAULT_TIMEOUT = 60",
                    "suggestion": "Update the documentation to state that the connection timeout defaults to 60 seconds.",
                    "confidence": "high",
                }
            ],
            "limitations": [
                "Assumption that DEFAULT_TIMEOUT directly controls the documented connection timeout."
            ],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(genuine_payload)),
        ):
            result = analyze_semantic_drift(default_diff, default_docs)

        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        self.assertEqual(finding["type"], "possible_semantic_drift")
        self.assertEqual(finding["confidence"], "high")
        self.assertTrue(finding["evidence_verified"])

    def test_genuine_behavior_change_finding_accepted(self) -> None:
        """
        When a real behavior change contradicts the documentation
        (e.g. a filter that silently drops items), the model should detect it.
        This mocks the expected correct finding.
        """
        filter_diff = """\
diff --git a/service.py b/service.py
--- a/service.py
+++ b/service.py
@@ -1,3 +1,3 @@
 def process_batch(items):
-    return [item.strip() for item in items]
+    return [item.strip() for item in items if len(item) > 3]
"""
        filter_docs = (
            "# Batch Processor\n\n"
            "Call `process_batch(items)` to sanitize a list of string elements.\n"
            "Returns all processed items with whitespace removed.\n"
        )
        behavior_payload = {
            "findings": [
                {
                    "type": "possible_semantic_drift",
                    "description": (
                        "The documentation states the function returns 'all processed items with whitespace removed', "
                        "but the code now filters out items with length <= 3, so not all items are returned."
                    ),
                    "evidence": "Returns all processed items with whitespace removed.\n-    return [item.strip() for item in items]\n+    return [item.strip() for item in items if len(item) > 3]",
                    "suggestion": "Update documentation to clarify that short items (length <= 3) are excluded.",
                    "confidence": "high",
                }
            ],
            "limitations": [],
        }
        with patch(
            "docdrift.ai_detector.urllib.request.urlopen",
            return_value=_urlopen_response(_envelope(behavior_payload)),
        ):
            result = analyze_semantic_drift(filter_diff, filter_docs)

        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        self.assertEqual(finding["type"], "possible_semantic_drift")
        # Evidence contains verbatim source text, should be verified
        self.assertTrue(finding["evidence_verified"])


if __name__ == "__main__":
    unittest.main()
