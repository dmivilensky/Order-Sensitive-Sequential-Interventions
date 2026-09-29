#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
( cd synthetic && python3 e1_support.py && python3 e2_planning.py && python3 e3_cube.py && python3 e4_scaling.py && python3 make_figures.py )
if [ -f "sepsis/data/Sepsis Cases - Event Log.xes" ]; then
  ( cd sepsis && python3 run_sepsis.py )
else
  echo "Sepsis log not found; see sepsis/data/README.txt. Skipping the real-data experiments."
fi
if [ -f "assistments/data/assistments_subset.csv" ]; then
  ( cd assistments && python3 run_assistments.py && python3 run_semisynthetic.py )
else
  echo "ASSISTments subset not found; see assistments/data/README.txt. Skipping."
fi
echo "done"
