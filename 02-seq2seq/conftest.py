"""Pytest config: put src/ on sys.path so tests (and the modules' own flat
imports like `from constants import ...`) resolve when running `pytest tests/`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
