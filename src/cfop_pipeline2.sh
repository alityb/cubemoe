#!/bin/bash
# Analysis-only CFOP pipeline: solve tests run in parallel on Modal; this downloads each result as it lands
# and runs the local stage/expert analysis. Skips anything already done. Log: out/prod/cfop_pipeline.log
cd /Users/alityb/projects/cubemoe; LOG=out/prod/cfop_pipeline.log; WS=dnfcubes
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
say "=== CFOP pipeline v2 (parallel solve tests) started"; deadline=$(( $(date +%s) + 8*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  left=0
  for s in 0 1 2 3 4; do
    tag=cfop_mixtral_6L256_s$s
    python3 -c "import json,sys;sys.exit(0 if 'CFOP' in json.load(open('out/prod/${tag}_analysis.json')) else 1)" 2>/dev/null && continue
    left=$((left+1))
    MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/${tag}_eval.json out/prod/${tag}_eval.json --force >/dev/null 2>&1 || continue
    [ -f out/prod/$tag.pt ] || MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/$tag.pt out/prod/$tag.pt --force >> "$LOG" 2>&1
    MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/$tag.json out/prod/$tag.json --force >/dev/null 2>&1
    say "cfop seed $s: solve test in; running stage/expert analysis"
    python3 -W ignore src/prod_cfop.py $tag >> "$LOG" 2>&1 && say "cfop seed $s: analysis done" || say "cfop seed $s: analysis FAILED"
  done
  [ $left -eq 0 ] && break
  sleep 120
done
git add src/cfop_pipeline2.sh src/prod_cfop.py out/prod/cfop_pipeline.log
for s in 0 1 2 3 4; do for f in out/prod/cfop_mixtral_6L256_s$s.json out/prod/cfop_mixtral_6L256_s${s}_eval.json out/prod/cfop_mixtral_6L256_s${s}_analysis.json; do [ -f "$f" ] && git add "$f"; done; done
git commit -q -m "CFOP production seeds: solve tests and stage/expert analysis" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> "$LOG" 2>&1
git push -q origin main >> "$LOG" 2>&1 && say "pushed" || say "push failed (committed locally)"
say "=== CFOP PIPELINE DONE"
