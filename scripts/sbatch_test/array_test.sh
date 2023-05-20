#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --time=1:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M


########## JOB NAME ##########

#SBATCH --job-name="test_testing"


########## MODULES ##########

set -e

module purge
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# output path
output="$scratch/output"
mkdir -p $output

# save path
save="/data/tsa/destevez/dennis/test_t_$SLURM_JOB_ID"
mkdir -p $save


########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/test/test.py $scratch
echo "end of copy"


########## ARRAY ##########

#SBATCH --array=1-10


########## RUN ##########

cd $scratch
echo "runing............"
N=$(sed -n -e "$SLURM_ARRAY_TASK_ID p")
echo "Current task ID: $SLURM_ARRAY_TASK_ID"
echo "Current job ID: $SLURM_ARRAY_JOB_ID"
echo "Current var : $N"
srun python3 test.py $SLURM_ARRAY_TASK_ID -p $output -i $N
echo "end of run"


########## SAVE ##########

echo "saving............"
cp -r $scratch $save
echo "end of save"


########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

