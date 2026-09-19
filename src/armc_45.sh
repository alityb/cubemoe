#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 4 5; do
  echo "=== state seed $s $(date +%T)"
  python3 -u src/train.py --model state --data mixed --epochs 3 --seed $s || exit 1
  python3 -u src/intervene_state.py --tag state_mixed_s$s --max_solves 800 || exit 1
  python3 -u src/contrast.py --tag state_mixed_s$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== ARM C 4,5 DONE $(date +%T)"
