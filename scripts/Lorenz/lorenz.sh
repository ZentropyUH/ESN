#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="lorenz"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10000M

#SBATCH --gres=gpu:A100:4

#SBATCH --time=1-00:00:00
#SBATCH --partition=graphic


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
save="/data/tsa/destevez/dennis/Lorenz_new/"
mkdir -p $save



########## COPY ##########

# TODO: Copy project
# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying data............"
cp -r /data/tsa/destevez/data/Lorenz/* $data
echo "end of copy"



########## RUN ##########

cd $ESN
echo "runing............"
srun python3 main.py grid -u 6000 -tl 20000 -fl 1000 -tr 1000 -d $data -o $output -i 1 -hp "./src/grid/combinations.json"
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
