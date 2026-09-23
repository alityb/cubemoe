#!/bin/bash
cd /Users/alityb/projects/cubemoe
while pgrep -f "sweep.py::finish_v1" > /dev/null; do sleep 60; done
sleep 20
./src/dl_v1.sh
if [ -f out/V1_top1_e8_local_s34.pt ]; then
  echo "=== ALL 5 V1 SEEDS PRESENT — running H_slack analysis $(date +%T)"
  python3 -u src/sweep_analysis.py
else
  echo "=== seed 34 checkpoint ABSENT after training exited — check out/finish_v1.log"
  tail -20 out/finish_v1.log
fi
