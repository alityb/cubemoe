#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 3 4 5; do
  echo "=== state seed $s $(date +%T)"
  python3 -u src/train.py --model state --data mixed --epochs 3 --seed $s || exit 1
  python3 -u src/contrast.py --tag state_mixed_s$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== ARM C SEEDS DONE $(date +%T)"
