#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "train.py --model" > /dev/null; do sleep 20; done
echo "=== refresh data cards $(date +%T)"
python3 -u src/qa.py --kind forced --nsample 1200 || exit 1
python3 -u src/qa.py --kind naive  --nsample 1200 || exit 1
echo "=== analysis $(date +%T)"
python3 -u src/run_analysis.py || exit 1
echo "=== figures $(date +%T)"
python3 -u src/figures.py || exit 1
echo "=== report $(date +%T)"
python3 -u src/report.py || exit 1
echo "=== ALL DONE $(date +%T)"
