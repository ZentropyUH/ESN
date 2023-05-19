#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=1:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=testing

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

# save path
save="/data/tsa/destevez/dennis/b_t_$SLURM_JOB_ID"


########## END ##########



########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/scripts/test_testing.sh $scratch
echo "end of copy"

########## END ##########





########## RUN ##########


echo "runing............"
sbatch $scratch/test.sh 57
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
