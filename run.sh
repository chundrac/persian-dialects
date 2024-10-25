#!/bin/bash
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

module load generic
module load singularity

srun singularity exec -u ~/anaconda_latest.sif bash invoker.sh "$@"
