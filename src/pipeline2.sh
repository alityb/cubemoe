#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "gen_data.py --kind forced" > /dev/null; do sleep 15; done
echo "=== QA forced $(date +%T)"
python3 -u src/qa.py --kind forced --nsample 1500
if [ $? -ne 0 ]; then echo "=== QA FAILED - refusing to train"; exit 1; fi
echo "=== QA PASSED - training $(date +%T)"
for spec in "moe forced 10" "dense forced 10" "hash forced 10" "state forced 3" "state_dense forced 3"; do
  set -- $spec
  echo "=== train $1 on $2 ($3 ep) $(date +%T)"
  python3 -u src/train.py --model $1 --data $2 --epochs $3
done
echo "=== PIPELINE DONE $(date +%T)"
