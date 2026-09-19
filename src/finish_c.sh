#!/bin/bash
cd /Users/alityb/projects/cubemoe
echo "=== state seed 2 $(date +%T)"
python3 -u src/train.py --model state --data mixed --epochs 3 --seed 2 || exit 1
echo "=== state_dense seed 2 $(date +%T)"
python3 -u src/train.py --model state_dense --data mixed --epochs 3 --seed 2 || exit 1
echo "=== eval C arm $(date +%T)"
for s in "" _s1 _s2; do
  python3 -u src/seedeval.py --tag state_mixed$s       --data mixed --state --max_solves 800 || exit 1
  python3 -u src/seedeval.py --tag state_dense_mixed$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== paired stats $(date +%T)"
python3 -u src/seedstats.py
echo "=== regenerate report $(date +%T)"
python3 -u src/report.py
echo "=== C ARM DONE $(date +%T)"
