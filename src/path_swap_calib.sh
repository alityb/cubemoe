#!/bin/bash
# Calibration for prereg #14 on the already-analysed held-out solves 0-1199 (thresholds are set from these, then confirmed on fresh solves).
cd /Users/alityb/projects/cubemoe; export SWAP_OUT=out/path_swap_calib
{ printf '%s cfop250k\n' cfop_mixtral_6L256_s{0..7}; printf '%s mixed250k\n' mixtral_6L256_s{0..4}; } | xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=3 python3 src/path_swap.py $0 $1 0 1200 >> out/path_swap_calib.log 2>&1 || echo "FAILED $0 $1" >> out/path_swap_calib.log'
echo "calib done $(date)" >> out/path_swap_calib.log
