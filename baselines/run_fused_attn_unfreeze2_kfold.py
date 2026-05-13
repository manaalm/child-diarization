"""Backwards-compat shim — moved to `encoders/run_fused_attn_unfreeze2_kfold.py`
per spec 022 US4 (2026-05-12). Old run-script invocations resolve via this shim for
one release cycle.

Run via:
    python -m encoders.run_fused_attn_unfreeze2_kfold [args...]
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("encoders.run_fused_attn_unfreeze2_kfold",
                     run_name="__main__", alter_sys=True)
