#!/bin/bash
# Overnight pipeline, independent of the Claude session. Progress: out/prod/overnight.log
# Result: out/prod/OVERNIGHT_REPORT.md (also committed + pushed at the end).
cd /Users/alityb/projects/cubemoe
LOG=out/prod/overnight.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
ws_of(){ case $1 in 0|1|2) echo thisaccisfortheschool;; 3|4) echo ali-moh-islam-1;; esac; }
report(){ python3 src/overnight_report.py >> "$LOG" 2>&1; }
say "=== overnight pipeline started"
MODAL_PROFILE=thisaccisfortheschool modal volume get phasesplit-vol /out/prod/mixtral_6L256_s0_eval.json out/prod/mixtral_6L256_s0_eval.json --force >> "$LOG" 2>&1
report
say "seed 0: stage A (model vs search) ..."
python3 -W ignore src/prod_stage_a.py mixtral_6L256_s0 >> "$LOG" 2>&1 && say "seed 0: stage A done" || say "seed 0: stage A FAILED"
report
say "seed 0: stages B-E (attention, features, experts, phase) ..."
python3 -W ignore src/prod_stages.py mixtral_6L256_s0 BCDE >> "$LOG" 2>&1 && say "seed 0: stages B-E done" || say "seed 0: stages B-E FAILED"
report
done_seeds=""; deadline=$(( $(date +%s) + 11*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for s in 1 2 3 4; do
    case " $done_seeds " in *" $s "*) continue;; esac
    ws=$(ws_of $s); tag=mixtral_6L256_s$s
    MODAL_PROFILE=$ws modal volume get phasesplit-vol /out/prod/$tag.json out/prod/$tag.json --force >/dev/null 2>&1 || continue
    fin=$(python3 -c "import json;g=json.load(open('out/prod/$tag.json'))['gates'];print(1 if g and g[-1]['step']>=58593 else 0)" 2>/dev/null)
    [ "$fin" = "1" ] || continue
    say "seed $s: training finished; downloading checkpoint"
    MODAL_PROFILE=$ws modal volume get phasesplit-vol /out/prod/$tag.pt out/prod/$tag.pt --force >> "$LOG" 2>&1
    for attempt in 1 2; do
      say "seed $s: 500-cube solve test on Modal ($ws), attempt $attempt"
      MODAL_PROFILE=$ws timeout 5400 modal run modal/prod.py::evaluate --tag $tag --n 500 --widths 64,256,1024 >> "$LOG" 2>&1
      MODAL_PROFILE=$ws modal volume get phasesplit-vol /out/prod/${tag}_eval.json out/prod/${tag}_eval.json --force >> "$LOG" 2>&1 && break
    done
    say "seed $s: phase analysis (stage E) ..."
    python3 -W ignore src/prod_stages.py $tag E >> "$LOG" 2>&1 && say "seed $s: stage E done" || say "seed $s: stage E FAILED"
    done_seeds="$done_seeds $s"; report
  done
  [ "$(echo $done_seeds | wc -w | tr -d ' ')" -ge 4 ] && break
  sleep 600
done
report
git add out/prod/OVERNIGHT_REPORT.md out/prod/overnight.log src/overnight.sh src/overnight_report.py src/prod_analysis.py src/prod_stages.py src/prod_stage_a.py modal/prod.py
for s in 0 1 2 3 4; do for f in out/prod/mixtral_6L256_s$s.json out/prod/mixtral_6L256_s${s}_eval.json out/prod/mixtral_6L256_s${s}_analysis.json; do [ -f "$f" ] && git add "$f"; done; done
git commit -q -m "Overnight: solve tests, model-vs-search, attention, features, experts and phase analysis for production-style seeds" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> "$LOG" 2>&1
git push -q origin main >> "$LOG" 2>&1 && say "pushed to GitHub" || say "git push failed (results are committed locally)"
say "=== ALL DONE. Read out/prod/OVERNIGHT_REPORT.md"
