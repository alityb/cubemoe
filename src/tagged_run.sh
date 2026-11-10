#!/bin/bash
# Driver for prereg #16 after training: gate on Modal, download checkpoints, local measures, report.
cd /Users/alityb/projects/cubemoe; mkdir -p out/tagged/swap
ws() { [ $1 -le 1 ] && echo ali-moh-islam-1 || echo dnfcubes; }
for s in 0 1 2 3; do MODAL_PROFILE=$(ws $s) modal run modal/prod.py::evaluate --tag cfop_tagged_6L256_s$s --n 200 --widths 256 --budget 90 --data cfop250k > out/tagged/gate_s$s.log 2>&1 & done; wait
for s in 0 1 2 3; do p=$(ws $s); MODAL_PROFILE=$p modal volume get --force phasesplit-vol out/prod/cfop_tagged_6L256_s${s}_eval.json out/prod/ >/dev/null
  MODAL_PROFILE=$p modal volume get --force phasesplit-vol out/prod/cfop_tagged_6L256_s$s.pt out/prod/ >/dev/null
  MODAL_PROFILE=$p modal volume get --force phasesplit-vol out/prod/cfop_tagged_6L256_s$s.json out/prod/ >/dev/null; done
grep -h "beam 256" out/tagged/gate_s*.log
for s in 0 1 2 3; do t=cfop_tagged_6L256_s$s
  python3 src/g1pll_analysis.py $t cfop250k >> out/tagged/measures.log 2>&1 || echo "FAILED g1pll $t" >> out/tagged/measures.log
  python3 src/geom_paths.py $t cfop250k >> out/tagged/measures.log 2>&1 || echo "FAILED geom $t" >> out/tagged/measures.log
  SWAP_OUT=out/tagged/swap python3 src/path_swap.py $t cfop250k 3600 1200 >> out/tagged/measures.log 2>&1 || echo "FAILED swap $t" >> out/tagged/measures.log
done
if grep -q FAILED out/tagged/measures.log; then echo "report not run: a job failed" >> out/tagged/measures.log; exit 1; fi
python3 src/tagged_report.py > out/tagged/report.log 2>&1; echo "report exit $?" >> out/tagged/measures.log
