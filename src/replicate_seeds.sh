#!/bin/bash
# Stages A-D (model vs search, attention, features, experts) on Kociemba production seeds 1-4.
cd /Users/alityb/projects/cubemoe; LOG=out/prod/replicate.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
for s in 1 2 3 4; do
  t=mixtral_6L256_s$s
  say "seed $s: stage A"; python3 -W ignore src/prod_stage_a.py $t >> "$LOG" 2>&1 && say "seed $s: A done" || say "seed $s: A FAILED"
  say "seed $s: stages B-D"; python3 -W ignore src/prod_stages.py $t BCD >> "$LOG" 2>&1 && say "seed $s: B-D done" || say "seed $s: B-D FAILED"
done
say "=== REPLICATION DONE"
