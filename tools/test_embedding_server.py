from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("embedding_server.py")
SPEC = importlib.util.spec_from_file_location("embedding_server", MODULE_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class EmbeddingServerTest(unittest.TestCase):
    def test_prefixes_are_asymmetric(self) -> None:
        self.assertEqual(server.prefixed(["текст"], "query"), ["search_query: текст"])
        self.assertEqual(server.prefixed(["текст"], "document"), ["search_document: текст"])

    def test_profiles_pin_berta_revision_and_pooling(self) -> None:
        profile = server.PROFILES["berta"]

        self.assertEqual(profile.revision, "914c8c8aed14042ed890fc2c662d5e9e66b2faa7")
        self.assertEqual(profile.pooling, "mean")
        self.assertEqual(profile.dimensions, 768)

    def test_frida_pins_revision_and_cls_pooling(self) -> None:
        profile = server.PROFILES["frida"]

        self.assertEqual(profile.revision, "aed004da8d09c33f6a51240fd9f5bcc625225b67")
        self.assertEqual(profile.pooling, "cls")
        self.assertEqual(profile.dimensions, 1536)


if __name__ == "__main__":
    unittest.main()
