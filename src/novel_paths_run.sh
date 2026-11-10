#!/bin/bash
# Driver for prereg #15: 13 models on novel CFOP states of held-out solves 4800-7199, 3 at a time, then the report.
cd /Users/alityb/projects/cubemoe; mkdir -p out/novel_paths
printf '%s\n' cfop_mixtral_6L256_s{0..7} mixtral_6L256_s{0..4} | xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=3 python3 src/novel_paths.py $0 >> out/novel_paths_run.log 2>&1 || echo "FAILED $0" >> out/novel_paths_run.log'
echo "all runs done $(date)" >> out/novel_paths_run.log
if grep -q FAILED out/novel_paths_run.log; then echo "report not run: a job failed" >> out/novel_paths_run.log; exit 1; fi
python3 src/novel_paths_report.py > out/novel_paths/report.log 2>&1; echo "report exit $?" >> out/novel_paths_run.log
