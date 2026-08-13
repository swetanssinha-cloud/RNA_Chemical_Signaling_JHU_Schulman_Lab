#!/bin/bash
#SBATCH --job-name=k_d_ds_sweep
#SBATCH --partition=shared               # shares a node instead of claiming a whole 48-core one
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION   # your PI/class allocation code from CoLdFront
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10               # 10 sweep values -> 1 core per sim, no batching needed
#SBATCH --mem=40G                        # ~4G/core, generous for an 11k-cell mesh
#SBATCH --time=00:30:00                  # ~7 min/sim x 1 batch + mesh build + overhead, with margin
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- environment -------------------------------------------------------
module load anaconda

# The FiPy solver shells out to a `gmsh` executable to check its version.
# That launcher has a `#!/usr/bin/env python` shebang, so the env's bin
# directory must be on PATH (not just `conda activate`-ed) or it resolves
# to a python without the gmsh module and fails with a misleading
# "Gmsh version must be >= 2.0" error.
source activate diffusion
export PATH="$CONDA_PREFIX/bin:$PATH"

# matplotlib must never try to open a window on a compute node
export MPLBACKEND=Agg

# --- sanity check before burning allocation hours -----------------------
python -c "import fipy, gmsh, pandas, matplotlib; print('env OK')"
gmsh --version

# --- run -----------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
srun python Parameter_sweep_unified.py
