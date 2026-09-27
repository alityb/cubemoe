#!/bin/bash
# Amendment 5 / H2b: position-decorrelated CFOP (cross_len=(0,25)), fresh seeds 50-54.
cd /Users/alityb/projects/cubemoe
for s in 50 51 52 53 54; do
  [ -f "out/moe_cfopdec_s$s.json" ] || { echo "=== moe seed $s $(date +%T)"; python3 -u src/train.py --model moe --data cfop_dec --epochs 10 --seed $s --tag moe_cfopdec_s$s || exit 1; }
done
for s in 50 51 52 53 54; do
  [ -f "out/hash_cfopdec_s$s.json" ] || { echo "=== hash seed $s $(date +%T)"; python3 -u src/train.py --model hash --data cfop_dec --epochs 10 --seed $s --tag hash_cfopdec_s$s || exit 1; }
done
echo "=== H2b TRAINING DONE $(date +%T)"
