#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="KS"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

#SBATCH --time=48:00:00
#SBATCH --partition=medium

#SBATCH --array=1-10



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

# project path
ESN=$scratch/ESN
mkdir -p $ESN

# output path
output="$scratch/output"
mkdir -p $output

# data path
data="$scratch/data"
mkdir -p $data

# save path
save="/data/tsa/destevez/dennis/KS/35_N64/run_$SLURM_ARRAY_TASK_ID"
mkdir -p $save



########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying data............"
cp -r /data/tsa/destevez/data/KS/35/N64/* $data
echo "end of copy"



########## RUN ##########

cd $ESN
echo "runing............"
srun python3 grid.py -o $output -d $data -i $SLURM_ARRAY_TASK_ID
echo "end of run"



########## SAVE ##########

echo "saving............"
cp -r $output/* $save
echo "end of save"



########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
