#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=1:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=short

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

########## END ##########





########## JOB NAME ##########

#SBATCH --job-name="short_test"

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

# save path
save="/data/tsa/destevez/dennis/b_s_$SLURM_JOB_ID"
mkdir -p $save

########## END ##########





########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/scripts/sbatch_test/test_short.sh $scratch
echo "end of copy"

########## END ##########





########## RUN ##########

cd $scratch
echo "runing............"
sbatch test_short.sh 57
echo "end of run"

########## END ##########





########## SAVE ##########

echo "saving............"
cp -r $scratch $save
echo "end of save"

########## END ##########





########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
