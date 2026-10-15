#!/bin/bash
# Waits for the CFOP production seeds on Modal (dnfcubes), then: download, 500-scramble solve test on Modal,
# CFOP stage/expert analysis locally. Independent of the Claude session. Log: out/prod/cfop_pipeline.log
cd /Users/alityb/projects/cubemoe; LOG=out/prod/cfop_pipeline.log; WS=dnfcubes
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
say "=== CFOP pipeline started"; done_seeds=""; deadline=$(( $(date +%s) + 14*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for s in 0 1 2 3 4; do
    case " $done_seeds " in *" $s "*) continue;; esac
    tag=cfop_mixtral_6L256_s$s
    MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/$tag.json out/prod/$tag.json --force >/dev/null 2>&1 || continue
    fin=$(python3 -c "import json;g=json.load(open('out/prod/$tag.json'))['gates'];print(1 if g and g[-1]['step']>=58593 else 0)" 2>/dev/null)
    [ "$fin" = "1" ] || continue
    say "cfop seed $s: training finished; downloading"
    MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/$tag.pt out/prod/$tag.pt --force >> "$LOG" 2>&1
    for attempt in 1 2; do
      say "cfop seed $s: 500-scramble solve test on Modal, attempt $attempt"
      MODAL_PROFILE=$WS timeout 7200 modal run modal/prod.py::evaluate --tag $tag --n 500 --widths 64,256,1024 --budget 90 --data cfop250k >> "$LOG" 2>&1
      MODAL_PROFILE=$WS modal volume get phasesplit-vol /out/prod/${tag}_eval.json out/prod/${tag}_eval.json --force >> "$LOG" 2>&1 && break
    done
    say "cfop seed $s: stage/expert analysis"
    python3 -W ignore src/prod_cfop.py $tag >> "$LOG" 2>&1 && say "cfop seed $s: analysis done" || say "cfop seed $s: analysis FAILED"
    done_seeds="$done_seeds $s"
  done
  [ "$(echo $done_seeds | wc -w | tr -d ' ')" -ge 5 ] && break
  sleep 600
done
git add src/cfop_pipeline.sh src/prod_cfop.py out/prod/cfop_pipeline.log
for s in 0 1 2 3 4; do for f in out/prod/cfop_mixtral_6L256_s$s.json out/prod/cfop_mixtral_6L256_s${s}_eval.json out/prod/cfop_mixtral_6L256_s${s}_analysis.json; do [ -f "$f" ] && git add "$f"; done; done
git commit -q -m "CFOP production seeds: solve tests and stage/expert analysis" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> "$LOG" 2>&1
git push -q origin main >> "$LOG" 2>&1 && say "pushed" || say "push failed (committed locally)"
say "=== CFOP PIPELINE DONE"
