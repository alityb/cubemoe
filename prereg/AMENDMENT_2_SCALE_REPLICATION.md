# AMENDMENT 2 — SCALE REPLICATION protocol

**Filed AFTER the 4L/d128 confirmatory results were known.** It therefore governs only a *future*
scale replication and adds no criterion to, and changes nothing about, the completed confirmatory
run. Kept in its own file so that `PREREGISTRATION.md` and `PREREG_AMENDMENT_1.md` remain exactly
as filed before results, with their hashes intact.

---

## Protocol

If a CUDA host becomes available, rerun **unchanged**:
`src/confirm_resume.sh` (train) -> `src/confirm.py` -> `src/confirm_gates.py`
with only these substitutions:
- **6 layers / d=256** (the `device_auto()=='cuda'` branch in `train.py` already selects this).
- **250k solves** (`python3 src/gen_data.py --kind mixed --n 250000`), re-run `src/qa.py --kind mixed`
  and require the same 16 hard gates to pass before training.
- **Seeds 20–24** per arm, plus hash twins 20–24 for arm A.

**No criterion changes.** L1–L4 thresholds, the last-layer rule, the strata, the `e1` definition, the
top-2 mechanics and the arm-C L1+L2+L4 rule all carry over verbatim from `PREREGISTRATION.md` and
A1.1–A1.4. Report pass/fail per leg in the same table. A scale replication that fails any leg is
reported as a failure at that scale, not reconciled against the 4L/d128 result.
