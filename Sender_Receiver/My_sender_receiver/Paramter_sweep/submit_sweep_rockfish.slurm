#!/bin/bash
#SBATCH --job-name=k_d_ds_sweep
#SBATCH --partition=parallel               # grabs individual nodes bc I'm greedy
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30               # 50 sweep values / 30 workers = 2 batches
#SBATCH --mem=120G                       # ~4G/core x 30 cores
#SBATCH --time=01:00:00                  # mesh cache was wiped by the last git pull, so this run
                                          # rebuilds every mesh from scratch -- generous margin
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- environment -------------------------------------------------------
module reset
module load anaconda3/2024.02-1
conda activate diffusion

# matplotlib must never try to open a window on a compute node
export MPLBACKEND=Agg

# --- sanity check before burning allocation hours -----------------------
python -c "import fipy, gmsh, pandas, matplotlib; print('env OK')"
gmsh --version

# --- run -----------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
srun python Parameter_sweep_unified.py
