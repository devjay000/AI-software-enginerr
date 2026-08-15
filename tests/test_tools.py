from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_engineer.config import Settings
from ai_engineer.tools import ToolPermissionError, ToolRegistry


class ToolPermissionTests(unittest.TestCase):
    def test_write_tools_require_human_approval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            registry = ToolRegistry(root, Settings(), approved=False)
            with self.assertRaises(ToolPermissionError):
                registry.invoke(
                    "apply_patch", diff="diff --git a/module.py b/module.py\n"
                )

    def test_read_file_rejects_outside_repository(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            registry = ToolRegistry(root, Settings())
            with self.assertRaises(PermissionError):
                registry.invoke("read_file", path="../secret.txt")

    def test_external_git_operations_require_approval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = ToolRegistry(root, Settings(), approved=False)
            with self.assertRaises(ToolPermissionError):
                registry.invoke("create_branch", branch="agent/fix")


if __name__ == "__main__":
    unittest.main()
