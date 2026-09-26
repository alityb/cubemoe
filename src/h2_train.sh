#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 40 41 42 43 44; do
  [ -f "out/moe_cfop_s$s.json" ] || { echo "=== moe seed $s $(date +%T)"; python3 -u src/train.py --model moe --data cfop --epochs 10 --seed $s --tag moe_cfop_s$s || exit 1; }
done
for s in 40 41 42 43 44; do
  [ -f "out/hash_cfop_s$s.json" ] || { echo "=== hash seed $s $(date +%T)"; python3 -u src/train.py --model hash --data cfop --epochs 10 --seed $s --tag hash_cfop_s$s || exit 1; }
done
echo "=== H2 TRAINING DONE $(date +%T)"
