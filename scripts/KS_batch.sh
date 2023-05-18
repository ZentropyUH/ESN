#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=48:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=medium

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

########## END ##########





########## JOB NAME ##########

#SBATCH --job-name="KS_batch"

########## END ##########




########## MODULES ##########

set -e

module purge
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

########## END ##########




########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# Script output
output="$scratch/output"
mkdir -p $output

# save path
save="/data/tsa/destevez/dennis/batch_$SLURM_JOB_ID"
mkdir -p $save

########## END ##########





########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/scripts/test.sh $scratch
echo "end of copy"

########## END ##########





########## RUN ##########

cd $ESN
echo "runing............"
sbatch $scratch/test.sh
echo "end of run"

########## END ##########





########## SAVE ##########

echo "saving............"
cp -r $output $save
echo "end of save"

########## END ##########





########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
