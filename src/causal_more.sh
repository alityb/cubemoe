#!/bin/bash
cd /Users/alityb/projects/cubemoe
echo "=== LAYER SWEEP on moe_mixed (is the causal effect depth-specific?) $(date +%T)"
for L in 0 1 2 3; do python3 -u src/intervene.py --tag moe_mixed --data mixed --max_solves 1200 --layer $L || exit 1; done
echo "=== CONTROL: hash twin (token-identity router) $(date +%T)"
python3 -u src/intervene.py --tag hash_mixed --data mixed --max_solves 1200 --layer 3 || exit 1
echo "=== naive-set model (seam sd 0.65) $(date +%T)"
python3 -u src/intervene.py --tag moe_naive --data naive --max_solves 1200 --layer 3 || exit 1
echo "=== CAUSAL EXTRAS DONE $(date +%T)"
