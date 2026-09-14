#!/bin/zsh
set -euo pipefail

cd /Users/harshitjoshi/rag_chunk_robustness
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

print "Resume started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
.venv/bin/python run_experiment.py
print "All configurations complete; starting final analysis."
.venv/bin/python analyze_results.py
.venv/bin/python make_figures.py
print "Final analysis complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
