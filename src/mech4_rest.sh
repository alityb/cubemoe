#!/bin/bash
# After seed 0's full-size exp4 finishes: seeds 1-4 at 400 solves, then the CFOP verification.
cd /Users/alityb/projects/cubemoe
while pgrep -f "mech4.py mixtral_6L256_s0" >/dev/null; do sleep 30; done
echo "[$(date '+%H:%M')] seed 0 exp4 finished" >> out/prod/mech4.log
for s in 1 2 3 4; do
  echo "[$(date '+%H:%M')] seed $s exp4 (400 solves)" >> out/prod/mech4.log
  python3 -u -W ignore src/mech4.py mixtral_6L256_s$s 400 >> out/prod/mech4.log 2>&1
done
echo "[$(date '+%H:%M')] EXP4 DONE" >> out/prod/mech4.log
python3 -u -W ignore src/cfop_verify.py 100 > out/prod/cfop_verify.log 2>&1
echo "[$(date '+%H:%M')] CFOP VERIFY DONE" >> out/prod/mech4.log
