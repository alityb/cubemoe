#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "train.py --model moe --data naive" > /dev/null; do sleep 10; done
for spec in "moe forced 10" "dense forced 10" "hash forced 10" "state forced 3" "state_dense forced 3"; do
  set -- $spec
  echo "=== $1 on $2 ($3 ep) $(date +%T)"
  python3 -u src/train.py --model $1 --data $2 --epochs $3
done
echo "=== ALL TRAINING DONE $(date +%T)"
