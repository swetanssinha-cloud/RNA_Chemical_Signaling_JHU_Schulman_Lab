#!/bin/bash
#SBATCH --job-name=build_diffusion_env
#SBATCH --partition=shared
#SBATCH --account=rschulm3_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load anaconda

# only installs mamba into the base env if it isn't already there
conda list -n base mamba 2>/dev/null | grep -q '^mamba ' || \
    conda install -n base -y -c conda-forge mamba

mamba env create -f Paramter_sweep/environment_rockfish.yml

echo "DONE building env"
