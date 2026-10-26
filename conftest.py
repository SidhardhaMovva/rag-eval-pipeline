# Ensures the repo root is on sys.path so `app` and `eval` import cleanly
# when pytest is run from the repository root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
