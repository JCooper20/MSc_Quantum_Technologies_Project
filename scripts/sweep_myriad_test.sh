#!/bin/bash -l
# Myriad SGE TEST array: 3 cheap cells (tasks 1-3 = N=64 at
# r = 0.2500, 0.2550, 0.2600), 4 trajectories each, short wall-time.
# Validates env, import stim, run, storage, logging end to end.
# Output goes to ~/Scratch/outputs_scaling/test/raw/ (kept separate so
# these small cells can NEVER be mistaken for complete production
# cells by the full array's skip logic).
# Submit with:            qsub scripts/sweep_myriad_test.sh
# BEFORE first submission: mkdir -p ~/Scratch/outputs_scaling/logs

#$ -l h_rt=0:30:00
#$ -l mem=4G
#$ -pe smp 4
#$ -t 1-3
#$ -N scaling_test
#$ -wd /home/zcapcoo/Scratch/outputs_scaling
#$ -o logs/
#$ -e logs/

set -e
module load python3/recommended
REPO="$HOME/MSc_Quantum_Technologies_Project"
cd "$REPO"
export PYTHONPATH=.
python -c "import stim, numpy; print('env OK: stim', stim.__version__, 'numpy', numpy.__version__)" \
  || { echo "FATAL: import stim failed — check 'pip install --user --only-binary :all: stim' under module python3/recommended"; exit 1; }
mkdir -p "$HOME/Scratch/outputs_scaling/test/raw" "$HOME/Scratch/outputs_scaling/logs"

export TEST_TRAJ=4
echo "TEST task $SGE_TASK_ID on $(hostname), NSLOTS=$NSLOTS, $(date)"
python scripts/run_cell_myriad.py
echo "TEST task $SGE_TASK_ID finished at $(date)"
