#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 1 2 3 4; do
  echo "=== moe   seed $s $(date +%T)"; python3 -u src/train.py --model moe   --data mixed --epochs 10 --seed $s || exit 1
  echo "=== dense seed $s $(date +%T)"; python3 -u src/train.py --model dense --data mixed --epochs 10 --seed $s || exit 1
done
for s in 1 2; do
  echo "=== state       seed $s $(date +%T)"; python3 -u src/train.py --model state       --data mixed --epochs 3 --seed $s || exit 1
  echo "=== state_dense seed $s $(date +%T)"; python3 -u src/train.py --model state_dense --data mixed --epochs 3 --seed $s || exit 1
done
echo "=== SEED TRAINING DONE $(date +%T)"
