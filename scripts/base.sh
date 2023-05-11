#!/bin/bash

########## EXECUTION TIME ##########

# TODO: max time
# Expected maximum runtime of job
#SBATCH --time=48:00:00

########## END ##########


########## RESOURCES TO USE ##########

# Number of processor cores (i.e. tasks)
#SBATCH --ntasks=1
#SBATCH --partition=graphic

# We will use 4 A100 GPUs
#SBATCH --gres=gpu:A100:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=2 # every task gets 2 CPUs
#SBATCH --mem-per-cpu=125000M

########## END ##########



########## EMAIL ##########

# Get notification to mail
#SBATCH --mail-type END

########## END ##########



########## JOB NAME ##########

# TODO: job name
#SBATCH --job-name="KS_CUDA"

########## END ##########




########## MODULES ##########

set -e

# load modules
module purge
module load cuda/11.4 # adjust cuda version as needed and add all other modules you used for development
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

########## END ##########




########## PATHS ##########

#Create your scratch space
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

# save path
save="/data/tsa/destevez/dennis/$SLURM_JOB_ID"
mkdir -p $save

########## END ##########





########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN $scratch

echo "copying data............"
cp -r /data/tsa/destevez/data $scratch/ESN
echo "end of copy\n"

########## END ##########





########## RUN ##########

cd $ESN
echo "runing............"
srun python3 $ESN/grid_script.py -o $output -d $data -n 1 -m 2
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