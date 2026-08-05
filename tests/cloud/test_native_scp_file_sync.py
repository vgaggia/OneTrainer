import subprocess
import unittest
from unittest.mock import Mock, patch

from modules.cloud.NativeSCPFileSync import NativeSCPFileSync


class NativeSCPFileSyncTest(unittest.TestCase):
    def test_failed_scp_uses_fallback(self):
        fallback = Mock()

        with patch(
            "modules.cloud.NativeSCPFileSync.subprocess.run",
            side_effect=subprocess.CalledProcessError(255, ["scp"]),
        ):
            NativeSCPFileSync._run_scp(["scp"], fallback)

        fallback.assert_called_once_with()

    def test_successful_scp_does_not_use_fallback(self):
        fallback = Mock()

        with patch("modules.cloud.NativeSCPFileSync.subprocess.run") as run:
            NativeSCPFileSync._run_scp(["scp"], fallback)

        run.assert_called_once_with(["scp"], check=True)
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
