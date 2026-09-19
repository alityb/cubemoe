#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "confirm_train.sh|train.py --model" > /dev/null; do sleep 30; done
echo "=== A1.1 hash-router twins, arm A seeds 10-14 $(date +%T)"
for s in 10 11 12 13 14; do
  python3 -u src/train.py --model hash --data mixed --epochs 10 --seed $s || exit 1
done
echo "=== CONFIRMATORY ANALYSIS $(date +%T)"
for s in 10 11 12 13 14; do python3 -u src/confirm.py --tag moe_mixed_s$s   --data mixed --max_solves 1200 || exit 1; done
for s in 10 11 12 13 14; do python3 -u src/confirm.py --tag state_mixed_s$s --data mixed --state --max_solves 800 || exit 1; done
echo "=== CONFIRMATORY GATES $(date +%T)"
python3 -u src/confirm_gates.py
echo "=== regenerate results.md $(date +%T)"
python3 -u src/report.py
echo "=== CONFIRMATORY DONE $(date +%T)"
