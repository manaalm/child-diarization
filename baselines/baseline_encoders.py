"""Backwards-compat shim — moved to `encoders/baseline_encoders.py` per spec 022 US4
(2026-05-12). Old import paths preserved for one release cycle. Update callers to
`from encoders.baseline_encoders import ...` at your convenience.

Supports both `import baselines.baseline_encoders` (re-exports the module) and
`python baselines/baseline_encoders.py [args...]` (runs the relocated __main__
via runpy).
"""
from encoders.baseline_encoders import *  # noqa: F401, F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("encoders.baseline_encoders", run_name="__main__", alter_sys=True)
