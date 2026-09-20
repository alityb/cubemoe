#!/bin/bash
cd /Users/alityb/projects/cubemoe
for s in 20 21 22 23 24; do
  for t in moe_mixed hash_mixed; do
    [ -f "out/${t}_s${s}.pt" ] || modal volume get phasesplit-vol out/${t}_s${s}.pt out/${t}_s${s}.pt >/dev/null 2>&1
    [ -f "out/${t}_s${s}.json" ] || modal volume get phasesplit-vol out/${t}_s${s}.json out/${t}_s${s}.json >/dev/null 2>&1
  done
  [ -f "out/state_mixed_s${s}.pt" ] || MODAL_PROFILE=dnfcubes modal volume get phasesplit-vol out/state_mixed_s${s}.pt out/state_mixed_s${s}.pt >/dev/null 2>&1
  [ -f "out/state_mixed_s${s}.json" ] || MODAL_PROFILE=dnfcubes modal volume get phasesplit-vol out/state_mixed_s${s}.json out/state_mixed_s${s}.json >/dev/null 2>&1
  echo "  seed $s done"
done
echo "DOWNLOAD COMPLETE"
