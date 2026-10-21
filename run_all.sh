#!/usr/bin/env bash
# Day 1: data, Jev calls (main rubric on both splits, alt rubric on test), baseline sweep, analysis.
# Day 2: ./run_all.sh stability
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "stability" ]]; then
  python jev_run.py --stability --run-tag day2
  python analyze.py
  exit 0
fi

python data.py
python jev_run.py                              # main rubric, train + test (feeds E1-E4)
python jev_run.py --rubric alt --split test    # wording sensitivity
python baselines.py                            # true labels + Jev labels (E3)
python analyze.py
python generalize.py
python mixture.py
python strong.py
python teacher_student.py
python teacher_student_paired.py
