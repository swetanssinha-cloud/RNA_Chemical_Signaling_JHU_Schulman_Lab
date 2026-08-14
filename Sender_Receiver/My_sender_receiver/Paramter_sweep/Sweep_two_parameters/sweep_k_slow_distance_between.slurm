#!/bin/bash
#SBATCH --job-name=k_slow_distance_between_sweep
#SBATCH --partition=parallel
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                 # 2 x 2 = 4 grid points -> 4 workers
#SBATCH --mem=20G
#SBATCH --time=00:45:00                   # includes up to 2 mesh builds (distance_between is geometric)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- environment -------------------------------------------------------
module reset
module load anaconda3/2024.02-1
conda activate diffusion

# matplotlib must never try to open a window on a compute node
export MPLBACKEND=Agg

# Without this, each of the Pool's worker processes independently spins up
# its own BLAS thread pool sized to all visible cores -- N processes x N
# threads each, all fighting over N physical cores. Pin every worker to a
# single thread so the process-level parallelism (the Pool) is the only
# parallelism in play.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- sanity check before burning allocation hours -----------------------
python -c "import fipy, gmsh, pandas, matplotlib; print('env OK')"
gmsh --version

# --- run -----------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
srun python sweep_k_slow_distance_between.py
