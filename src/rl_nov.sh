#!/bin/bash
# Seeds 1-2 of prereg #7 (RL routing) and #8 (opening swap), unchanged protocol. Stages run in order; each waits until
# its outputs exist on the Modal volume (a running container cannot see files committed after it started).
cd /Users/alityb/projects/cubemoe; LOG=out/rl/nov.log; export MODAL_PROFILE=ali-moh-islam-1
T1=cfop_mixtral_6L256_s1; T2=cfop_mixtral_6L256_s2
have() { for f in "$@"; do modal volume ls phasesplit-vol out/rl 2>/dev/null | grep -q "out/rl/$f$" || return 1; done; }
stage() { local name="$1" jobs="$2"; shift 2
  echo "[$(date '+%m-%d %H:%M')] stage $name" >> $LOG
  have "$@" || modal run --detach modal/prod.py::scripts --jobs "$jobs" >> $LOG 2>&1
  until have "$@"; do sleep 120; done; echo "[$(date '+%m-%d %H:%M')] stage $name done" >> $LOG; }
stage bins "rl_eval.py bins $T1; rl_eval.py bins $T2" bins_$T1.json bins_$T2.json
J=""; OUTS=""
for T in $T1 $T2; do
  J="$J rl_eval.py eval $T base;"; OUTS="$OUTS eval_${T}_base.json"
  for a in "router 3e-4 0.0003" "lora 3e-3 0.003" "experts 1e-4 0.0001" "full 1e-4 0.0001"; do set -- $a
    J="$J rl_confirm.py $T $1 $2 0;"; OUTS="$OUTS mech_${T}_$1_lr$3_s0.json"; done
done
stage train_eval "$J" $OUTS
stage swap "rl_swap.py $T1 ${T1}_full_lr0.0001_s0 ${T1}_experts_lr0.0001_s0; rl_swap.py $T2 ${T2}_full_lr0.0001_s0 ${T2}_experts_lr0.0001_s0" \
  swap_${T1}_full_lr0.0001_s0.json swap_${T1}_experts_lr0.0001_s0.json swap_${T2}_full_lr0.0001_s0.json swap_${T2}_experts_lr0.0001_s0.json
for f in $(modal volume ls phasesplit-vol out/rl 2>/dev/null | grep -E "_s[12]_|_s[12]\.json"); do modal volume get --force phasesplit-vol $f out/rl/ >/dev/null 2>&1; done
echo "[$(date '+%m-%d %H:%M')] ALL DONE" >> $LOG
