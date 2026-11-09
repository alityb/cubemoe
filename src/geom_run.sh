#!/bin/bash
# Driver for prereg #12: 13 models x (own positions, other family's positions), 3 at a time, then the report.
cd /Users/alityb/projects/cubemoe; mkdir -p out/geom
jobs=()
for s in 0 1 2 3 4; do jobs+=("mixtral_6L256_s$s mixed250k" "mixtral_6L256_s$s cfop250k"); done
for s in 0 1 2 3 4 5 6 7; do jobs+=("cfop_mixtral_6L256_s$s cfop250k" "cfop_mixtral_6L256_s$s mixed250k"); done
printf '%s\n' "${jobs[@]}" | xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=3 python3 src/geom_paths.py $0 $1 >> out/geom_run.log 2>&1 || echo "FAILED $0 $1" >> out/geom_run.log'
echo "all runs done $(date)" >> out/geom_run.log
python3 src/geom_report.py > out/geom/report.log 2>&1; echo "report exit $?" >> out/geom_run.log
