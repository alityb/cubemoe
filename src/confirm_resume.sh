#!/bin/bash
# Idempotent: skips any model whose training record already exists, so a teardown
# costs only the in-flight model rather than the whole queue. Safe to re-run.
cd /Users/alityb/projects/cubemoe
run_if_missing () {  # $1=model $2=seed $3=epochs $4=tagprefix
  if [ -f "out/$4_s$2.json" ]; then echo "=== SKIP $4_s$2 (already trained)"; return 0; fi
  echo "=== TRAIN $4_s$2 $(date +%T)"
  python3 -u src/train.py --model $1 --data mixed --epochs $3 --seed $2 || exit 1
}
for s in 10 11 12 13 14; do run_if_missing moe   $s 10 moe_mixed;   done
for s in 10 11 12 13 14; do run_if_missing state $s 3  state_mixed; done
for s in 10 11 12 13 14; do run_if_missing hash  $s 10 hash_mixed;  done

echo "=== CONFIRMATORY ANALYSIS $(date +%T)"
for s in 10 11 12 13 14; do
  [ -f "out/confirm/moe_mixed_s$s.json" ] || python3 -u src/confirm.py --tag moe_mixed_s$s --data mixed --max_solves 1200 || exit 1
done
for s in 10 11 12 13 14; do
  [ -f "out/confirm/state_mixed_s$s.json" ] || python3 -u src/confirm.py --tag state_mixed_s$s --data mixed --state --max_solves 800 || exit 1
done
echo "=== CONFIRMATORY GATES $(date +%T)"
python3 -u src/confirm_gates.py
echo "=== regenerate results.md $(date +%T)"
python3 -u src/report.py
echo "=== CONFIRMATORY DONE $(date +%T)"
