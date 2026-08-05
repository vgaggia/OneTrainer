import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from modules.util.path_util import write_json_atomic


class WriteJsonAtomicTest(unittest.TestCase):
    def test_retries_transient_windows_replace_error(self):
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "config.json"
            destination.write_text('{"old": true}', encoding="utf-8")
            real_replace = os.replace
            attempts = 0

            def replace_after_transient_lock(source, target):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError(5, "Access is denied")
                real_replace(source, target)

            with (
                patch("modules.util.path_util.os.replace", side_effect=replace_after_transient_lock),
                patch("modules.util.path_util.time.sleep") as sleep,
            ):
                write_json_atomic(str(destination), {"new": True})

            self.assertEqual(2, attempts)
            sleep.assert_called_once_with(0.05)
            self.assertIn('"new": true', destination.read_text(encoding="utf-8"))
            self.assertEqual([], list(Path(temp_dir).glob("*.write")))


if __name__ == "__main__":
    unittest.main()
