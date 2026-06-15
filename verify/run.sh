#!/usr/bin/env bash
# Verify the demoniC port against the upstream Python reference (out-of-band:
# the .dmc carries no test scaffolding the original lacked). Set DMC to the dmc
# binary, or have `dmc` on PATH.
set -euo pipefail
DMC="${DMC:-dmc}"
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$root/verify/verify_port.py" --mode floats \
  --ref python3 "$root/adamw_step.py" \
  --dmc $DMC run "$root/adamw_step.dmc"
