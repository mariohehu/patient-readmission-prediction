"""Make the project importable as `from src.X import ...` during pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
