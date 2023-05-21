#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="test_testing"
#SBATCH --array=1-10


#SBATCH --time=1:00:00
#SBATCH --partition=testing

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000M



########## MODULES ##########

set -e

module purge
module load python/3.10.5


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
cp -r /data/tsa/destevez/dennis/ESN/src/grid/combinations.txt $scratch
echo "end of copy"



########## RUN ##########

cd $scratch
echo "runing............"
srun python3 test.py -a $SLURM_ARRAY_TASK_ID
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

