#!/bin/bash
# Experiment 4 on all five Kociemba seeds, after the local CFOP analysis finishes.
cd /Users/alityb/projects/cubemoe
while pgrep -f cfop_local.sh >/dev/null; do sleep 30; done
for s in 0 1 2 3 4; do
  echo "[$(date '+%H:%M')] seed $s exp4" >> out/prod/mech4.log
  python3 -W ignore src/mech4.py mixtral_6L256_s$s >> out/prod/mech4.log 2>&1
done
echo "[$(date '+%H:%M')] EXP4 DONE" >> out/prod/mech4.log
