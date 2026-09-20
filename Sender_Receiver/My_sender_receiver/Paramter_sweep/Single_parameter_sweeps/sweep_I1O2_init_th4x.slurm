#!/bin/bash
#SBATCH --job-name=I1O2_init_th4x_sweep
#SBATCH --partition=parallel
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11                 # 11 sweep values -> 11 workers
#SBATCH --mem=20G
#SBATCH --time=00:30:00
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
# Runs sweep_I1O2_init_th4x.py, NOT sweep_I1O2_init.py -- it points at a
# fresh output_dir (sweep_I1O2_init_th4x_kp=0.01_ImprovedV4_5mmx5mm/)
# instead of the original sweep_I1O2_init_kp=0.01_ImprovedV4_5mmx5mm/, which
# is still full of pre-fix (Th2_init fixed at 0.4, not scaled to I2_init)
# results. Reusing that folder would make every run look like a no-op SKIP
# instead of actually re-running with the new Th2_init = 4 * I2_init physics.
cd "$SLURM_SUBMIT_DIR"
srun python sweep_I1O2_init_th4x.py
