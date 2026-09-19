#!/bin/bash
cd /Users/alityb/projects/cubemoe
echo "=== gen mixed $(date +%T)"
python3 -u src/gen_data.py --kind mixed --n 25000 --workers 10 || exit 1
echo "=== QA mixed $(date +%T)"
python3 -u src/qa.py --kind mixed --nsample 1200 || { echo "=== QA FAILED - refusing to train"; exit 1; }
echo "=== QA PASSED - training $(date +%T)"
for spec in "moe mixed 10" "dense mixed 10" "hash mixed 10" "state mixed 3" "state_dense mixed 3"; do
  set -- $spec
  echo "=== train $1 on $2 ($3 ep) $(date +%T)"
  python3 -u src/train.py --model $1 --data $2 --epochs $3 || exit 1
done
echo "=== analysis $(date +%T)"
python3 -u src/run_analysis.py || exit 1
echo "=== figures $(date +%T)"
python3 -u src/figures.py || exit 1
echo "=== report $(date +%T)"
python3 -u src/report.py || exit 1
echo "=== ALL DONE $(date +%T)"
