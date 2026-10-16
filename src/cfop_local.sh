#!/bin/bash
# CFOP stage/expert analysis for all 5 models, locally (no Modal). Log: out/prod/cfop_pipeline.log
cd /Users/alityb/projects/cubemoe; LOG=out/prod/cfop_pipeline.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
say "=== CFOP local analysis started (Modal workspace dnfcubes hit its \$40 limit; solve tests stopped)"
for s in 0 1 2 3 4; do
  t=cfop_mixtral_6L256_s$s
  python3 -c "import json,sys;sys.exit(0 if 'CFOP' in json.load(open('out/prod/${t}_analysis.json')) else 1)" 2>/dev/null && { say "seed $s already done"; continue; }
  say "cfop seed $s: stage/expert analysis"; python3 -W ignore src/prod_cfop.py $t >> "$LOG" 2>&1 && say "cfop seed $s: done" || say "cfop seed $s: FAILED"
done
say "=== CFOP LOCAL ANALYSIS DONE"
