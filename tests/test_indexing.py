from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_engineer.config import Settings
from ai_engineer.indexing import CodeIndexer


class IndexingTests(unittest.TestCase):
    def test_python_ast_symbols_and_imports_are_extracted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "service.py").write_text(
                "import os\nfrom package.auth import session\n\nclass UserService:\n    def authenticate(self, token):\n        return bool(token)\n\ndef test_authenticate():\n    assert UserService().authenticate('x')\n",
                encoding="utf-8",
            )
            result = CodeIndexer(Settings()).index(root)
            symbols = {chunk.symbol: chunk for chunk in result.chunks}
            self.assertIn("UserService", symbols)
            self.assertIn("UserService.authenticate", symbols)
            self.assertEqual(symbols["UserService.authenticate"].kind, "method")
            self.assertIn("package.auth.session", result.dependencies["service.py"])

    def test_secret_containing_source_is_excluded(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unsafe.py").write_text(
                "token = 'abcdefghijklmnopqrstuvwxyz'\n", encoding="utf-8"
            )
            result = CodeIndexer(Settings()).index(root)
            self.assertFalse(result.chunks)
            self.assertIn("unsafe.py", result.excluded_files)


if __name__ == "__main__":
    unittest.main()
