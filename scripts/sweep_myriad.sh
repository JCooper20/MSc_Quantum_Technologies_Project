#!/bin/bash -l
# Myriad SGE array jobscript: full finite-size scaling sweep.
# One array task = one (N, r) cell (mapping in scripts/sweep_params.txt).
# Submit from anywhere with:   qsub scripts/sweep_myriad.sh
# BEFORE first submission:     mkdir -p ~/Scratch/outputs_scaling/logs

# ---- resources (edit here) -------------------------------------------
#$ -l h_rt=8:00:00
#$ -l mem=4G
#$ -pe smp 16
#$ -t 1-459
#$ -tc 36
#$ -N scaling_sweep
#$ -wd /home/zcapcoo/Scratch/outputs_scaling
#$ -o logs/
#$ -e logs/

set -e

# ---- environment ------------------------------------------------------
module load python3/recommended
# Stim is a --user install (pip install --user --only-binary :all: stim)
# because building from source fails on Myriad's compilers; the module's
# python picks up ~/.local site-packages automatically.

# Repo location (ASSUMPTION: cloned to $HOME — edit if elsewhere)
REPO="$HOME/MSc_Quantum_Technologies_Project"
cd "$REPO"
export PYTHONPATH=.

# fail fast, loudly, if the environment is broken
python -c "import stim, numpy; print('env OK: stim', stim.__version__, 'numpy', numpy.__version__)" \
  || { echo "FATAL: import stim failed — check 'pip install --user --only-binary :all: stim' under module python3/recommended"; exit 1; }

mkdir -p "$HOME/Scratch/outputs_scaling/raw" "$HOME/Scratch/outputs_scaling/logs"

echo "task $SGE_TASK_ID starting on $(hostname) with NSLOTS=$NSLOTS at $(date)"
python scripts/run_cell_myriad.py
echo "task $SGE_TASK_ID finished at $(date)"
