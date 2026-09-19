#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 10 11 12 13 14; do
  echo "=== CONF moe seed $s $(date +%T)"
  python3 -u src/train.py --model moe --data mixed --epochs 10 --seed $s || exit 1
done
for s in 10 11 12 13 14; do
  echo "=== CONF state seed $s $(date +%T)"
  python3 -u src/train.py --model state --data mixed --epochs 3 --seed $s || exit 1
done
echo "=== CONFIRMATORY TRAINING DONE $(date +%T)"
