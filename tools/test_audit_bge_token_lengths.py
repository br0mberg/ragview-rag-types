import importlib.util
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_bge_token_lengths.py")
SPEC = importlib.util.spec_from_file_location("audit_bge_token_lengths", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditBgeTokenLengthsTest(unittest.TestCase):

    def test_indexable_text_matches_java_contract(self):
        self.assertEqual(
            "Title. Body",
            MODULE.indexable_text({"title": " Title ", "text": " Body "}),
        )
        self.assertEqual(
            "Title. Body",
            MODULE.indexable_text({"title": "Title", "text": "Title. Body"}),
        )

    def test_summary_counts_both_limits(self):
        self.assertEqual(
            {"pairs": 5, "over512": 3, "over1024": 1, "min": 100, "max": 1025},
            MODULE.summarize([512, 100, 513, 1024, 1025]),
        )

    def test_read_jsonl_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id":"ok"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"rows\.jsonl:2"):
                MODULE.read_jsonl(path)

    def test_require_unique_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            MODULE.require_unique([{"id": "same"}, {"id": "same"}], "id", "rows")

    def test_verify_summary_recomputes_counts_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, lengths in (("candidates.csv", [512, 513]), ("targets.csv", [1025])):
                with (root / filename).open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=["pair_tokens"])
                    writer.writeheader()
                    writer.writerows({"pair_tokens": length} for length in lengths)
            summary = {
                "artifacts": {
                    "candidatePairTokenLengths": {
                        "filename": "candidates.csv",
                        "sha256": hashlib.sha256((root / "candidates.csv").read_bytes()).hexdigest(),
                    },
                    "targetPairTokenLengths": {
                        "filename": "targets.csv",
                        "sha256": hashlib.sha256((root / "targets.csv").read_bytes()).hexdigest(),
                    },
                },
                "candidatePairs": MODULE.summarize([512, 513]),
                "targetPairs": MODULE.summarize([1025]),
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            MODULE.verify_summary(summary_path)

            with (root / "targets.csv").open("a", encoding="utf-8") as output:
                output.write("2\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.verify_summary(summary_path)


if __name__ == "__main__":
    unittest.main()
