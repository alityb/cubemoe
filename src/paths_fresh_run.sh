#!/bin/bash
# Driver for prereg #13: 16 models (8 CFOP, 5 Kociemba, 3 untrained) on the fresh CFOP positions, 3 at a time, then the report.
cd /Users/alityb/projects/cubemoe; mkdir -p out/paths_fresh
printf '%s\n' cfop_mixtral_6L256_s{0..7} mixtral_6L256_s{0..4} random_s{100,101,102} | xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=3 python3 src/paths_fresh.py $0 >> out/paths_fresh_run.log 2>&1 || echo "FAILED $0" >> out/paths_fresh_run.log'
echo "all runs done $(date)" >> out/paths_fresh_run.log
if grep -q FAILED out/paths_fresh_run.log; then echo "report not run: a job failed" >> out/paths_fresh_run.log; exit 1; fi
python3 src/paths_fresh_report.py > out/paths_fresh/report.log 2>&1; echo "report exit $?" >> out/paths_fresh_run.log
