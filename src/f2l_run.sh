#!/bin/bash
# Driver for prereg #11 on dnfcubes: wait for training of seeds 5-7, then gate, then measures, then the verdict.
cd /Users/alityb/projects/cubemoe; export MODAL_PROFILE=dnfcubes
while MODAL_PROFILE=dnfcubes modal app list 2>/dev/null | grep -q "ephemeral"; do sleep 120; done
echo "training done $(date)"
for s in 5 6 7; do modal run modal/prod.py::evaluate --tag cfop_mixtral_6L256_s$s --n 200 --widths 256 --budget 90 --data cfop250k > out/modal_f2l_gate_s$s.log 2>&1 & done; wait
grep -h "beam 256" out/modal_f2l_gate_s*.log
for s in 5 6 7; do modal volume get --force phasesplit-vol out/prod/cfop_mixtral_6L256_s${s}_eval.json out/prod/ >/dev/null; modal volume get --force phasesplit-vol out/prod/cfop_mixtral_6L256_s$s.pt out/prod/ >/dev/null; done
modal run modal/prod.py::scripts --jobs "f2l_split.py cfop_mixtral_6L256_s5;f2l_split.py cfop_mixtral_6L256_s6;f2l_split.py cfop_mixtral_6L256_s7" > out/modal_f2l_test.log 2>&1
for s in 5 6 7; do modal volume get --force phasesplit-vol out/f2l/cfop_mixtral_6L256_s$s.json out/f2l/ >/dev/null; done
python3 src/f2l_report.py
