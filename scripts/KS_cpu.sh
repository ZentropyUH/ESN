#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=48:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=long

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25000M

########## END ##########





########## JOB NAME ##########

#SBATCH --job-name="KS_cpu"

########## END ##########




########## MODULES ##########

set -e

module purge
module load cuda/11.4
module load python/3.10.5

########## END ##########




########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# Script output
output="$scratch/output"
mkdir -p $output

# project path
ESN="$scratch/ESN"

# data path
data="$ESN/data"
mkdir -p $data

# save path
save="/data/tsa/destevez/dennis/$SLURM_JOB_ID"
mkdir -p $save

########## END ##########





########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN $scratch

echo "copying data............"
cp -r /data/tsa/destevez/data/KS/35/N64/* $data
echo "end of copy\n"

########## END ##########





########## RUN ##########

cd $ESN
echo "runing............"
srun python3 $ESN/grid.py -o $output -d $data -n 1 -m 2
echo "end of run\n"

########## END ##########





########## SAVE ##########

echo "saving............"
cp -r $output $save
echo "end of save\n"

########## END ##########





########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
