import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `import synth` works in all tests.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
