#!/bin/bash
# Re-run H1 arm-A correlational legs with the FIXED 53+t extraction (MEMORY.md §14).
# Writes to confirm_fixed/ and confirm_scale_fixed/ so the published numbers stay auditable.
cd /Users/alityb/projects/cubemoe
for s in 10 11 12 13 14; do
  [ -f "out/confirm_fixed/moe_mixed_s$s.json" ] || { echo "=== s$s (25k) $(date +%T)"; python3 -u src/confirm.py --tag moe_mixed_s$s --data mixed --max_solves 1200 --outdir confirm_fixed || exit 1; }
done
for s in 20 21 22 23 24; do
  [ -f "out/confirm_scale_fixed/moe_mixed_s$s.json" ] || { echo "=== s$s (250k) $(date +%T)"; python3 -u src/confirm.py --tag moe_mixed_s$s --data mixed250k --max_solves 1200 --outdir confirm_scale_fixed || exit 1; }
done
echo "=== H1 RERUN DONE $(date +%T)"
