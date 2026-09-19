#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "seeds.sh|train.py --model" > /dev/null; do sleep 30; done
echo "=== seed eval $(date +%T)  (matched held-out: 1200 solves seq arm, 800 state arm)"
for s in "" _s1 _s2 _s3 _s4; do
  python3 -u src/seedeval.py --tag moe_mixed$s   --data mixed --max_solves 1200 || exit 1
  python3 -u src/seedeval.py --tag dense_mixed$s --data mixed --max_solves 1200 || exit 1
done
for s in "" _s1 _s2; do
  python3 -u src/seedeval.py --tag state_mixed$s       --data mixed --state --max_solves 800 || exit 1
  python3 -u src/seedeval.py --tag state_dense_mixed$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== paired stats $(date +%T)"
python3 -u src/seedstats.py
echo "=== SEED STUDY DONE $(date +%T)"
