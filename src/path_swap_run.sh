#!/bin/bash
# Driver for prereg #14 (confirmation): 13 models on held-out solves 3600-4799, 3 at a time, then the report.
cd /Users/alityb/projects/cubemoe; mkdir -p out/path_swap; export SWAP_OUT=out/path_swap
{ printf '%s cfop250k\n' cfop_mixtral_6L256_s{0..7}; printf '%s mixed250k\n' mixtral_6L256_s{0..4}; } | xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=3 python3 src/path_swap.py $0 $1 3600 1200 >> out/path_swap_run.log 2>&1 || echo "FAILED $0 $1" >> out/path_swap_run.log'
echo "all runs done $(date)" >> out/path_swap_run.log
if grep -q FAILED out/path_swap_run.log; then echo "report not run: a job failed" >> out/path_swap_run.log; exit 1; fi
python3 src/path_swap_report.py > out/path_swap/report.log 2>&1; echo "report exit $?" >> out/path_swap_run.log
