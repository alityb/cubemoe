#!/bin/bash
# EXPLORATORY: state-only (no move history, no position) CFOP models, same recipe as Kociemba arm C.
cd /Users/alityb/projects/cubemoe
for s in 60 61 62 63 64; do
  [ -f "out/state_cfop_s$s.json" ] || { echo "=== state_cfop seed $s $(date +%T)"; python3 -u src/train.py --model state --data cfop --epochs 3 --seed $s --tag state_cfop_s$s || exit 1; }
done
echo "=== CFOP STATE TRAINING DONE $(date +%T)"
