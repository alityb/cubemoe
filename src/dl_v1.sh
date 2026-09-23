#!/bin/bash
cd /Users/alityb/projects/cubemoe
export MODAL_PROFILE=thisaccisfortheschool
for s in 30 31 32 33 34; do
  t=V1_top1_e8_local_s$s
  [ -f "out/$t.pt" ] || modal volume get phasesplit-vol out/$t.pt out/$t.pt >/dev/null 2>&1
  [ -f "out/$t.json" ] || modal volume get phasesplit-vol out/$t.json out/$t.json >/dev/null 2>&1
  [ -f "out/$t.pt" ] && echo "  have $t" || echo "  pending $t"
done
