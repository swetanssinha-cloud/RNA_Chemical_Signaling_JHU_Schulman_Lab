#!/bin/bash
#SBATCH --job-name=k_d_ds_sweep
#SBATCH --partition=parallel               # grabs individual nodes bc I'm greedy
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30               # 10 sweep values + 1 spare core (Pool uses cpu_count()-1 workers)
#SBATCH --mem=44G                        # ~4G/core, generous for an 11k-cell mesh
#SBATCH --time=00:20:00                  # ~7 min/sim x 1 batch + mesh build + overhead, with margin
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
