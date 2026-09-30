"""
Pytest configuration.

Putting the project root on `sys.path` here means `import backend...` works in
tests whether pytest is invoked from the repository root or from `tests/`, and
without requiring an editable install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
