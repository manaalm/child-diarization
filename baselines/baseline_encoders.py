"""Backwards-compat shim — moved to `encoders/baseline_encoders.py` per spec 022 US4
(2026-05-12). Old import paths preserved for one release cycle. Update callers to
`from encoders.baseline_encoders import ...` at your convenience.
"""
from encoders.baseline_encoders import *  # noqa: F401, F403
