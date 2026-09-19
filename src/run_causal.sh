#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in "" _s1 _s2 _s3 _s4; do
  echo "=== causal moe_mixed$s $(date +%T)"
  python3 -u src/intervene.py --tag moe_mixed$s --data mixed --max_solves 1200 || exit 1
done
echo "=== CAUSAL DONE $(date +%T)"
