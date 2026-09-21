#!/bin/bash
#SBATCH --job-name=distance_x_I1O2_sweep
#SBATCH --partition=parallel
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # 26 x 11 = 286 grid points -> 32 workers, 9 batches
#SBATCH --mem=128G                          # 4G/worker, matching sweep_k_d_ds_k_d_ss.slurm's ratio
#SBATCH --time=03:00:00                   # includes ~19 fresh mesh builds (distance_between is
                                             # geometric; only 7 of 26 values have a cached mesh at
                                             # the current 5000x5000 domain size) plus 286 sims;
                                             # resume/skip picks up if this times out
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
srun python sweep_distance_between_x_I1O2_init.py
