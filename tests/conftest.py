"""Pytest config — make the project root importable so tests can do
`import agent` etc. without an editable install."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
