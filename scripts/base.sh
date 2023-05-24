#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="base"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M

#SBATCH --time=48:00:00
#SBATCH --partition=medium



########## MODULES ##########

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch


########## COPY ##########


rm -rf /data/tsa/destevez/dennis/Lorenz


########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
