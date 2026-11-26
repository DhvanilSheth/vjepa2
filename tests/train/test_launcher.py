# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""app/main.py must report a failed run through its exit status.

It used to start its worker processes and return 0 whatever happened to them,
so `for arm in ...; do python -m app.main ...; done` carried on after a crash
(ladder audit L-10).
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLauncherExitStatus(unittest.TestCase):
    def test_a_crashed_rank_makes_the_launcher_exit_non_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "app.main", "--fname", "/nonexistent/config.yaml", "--devices", "cpu"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertIn("FileNotFoundError", result.stderr)
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
