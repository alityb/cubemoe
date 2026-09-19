#!/bin/bash
cd /Users/alityb/projects/cubemoe
echo "=== contrast: sequence arm (router vs nulls) $(date +%T)"
for s in "" _s1 _s2 _s3 _s4; do
  python3 -u src/contrast.py --tag moe_mixed$s --data mixed --max_solves 1200 || exit 1
done
echo "=== contrast: position-blind arm C $(date +%T)"
for s in "" _s1 _s2; do
  python3 -u src/contrast.py --tag state_mixed$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== contrast: hash twin (null) $(date +%T)"
python3 -u src/contrast.py --tag hash_mixed --data mixed --max_solves 1200 || exit 1
echo "=== contrast: dense sanity check $(date +%T)"
python3 -u src/contrast.py --tag dense_mixed --data mixed --max_solves 1200 || exit 1
python3 -u src/contrast.py --tag state_dense_mixed --data mixed --state --max_solves 800 || exit 1
echo "=== CONTRAST DONE $(date +%T)"
