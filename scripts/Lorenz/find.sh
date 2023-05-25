#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="lorenz"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10000M

#SBATCH --time=14-00:00:00
#SBATCH --partition=long



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
save="/data/tsa/destevez/dennis/results/Lorenz/"
mkdir -p $save



########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying data............"
# cp -r /data/tsa/destevez/dennis/Lorenz/* $data
cp -r /data/tsa/destevez/dennis/Lorenz/0.4_8_0.0001_0.91_0.0 $data
cp -r /data/tsa/destevez/dennis/Lorenz/0.4_8_0.0001_0.91_1.0 $data
cp -r /data/tsa/destevez/dennis/Lorenz/0.4_8_0.0001_0.95_0.0 $data
cp -r /data/tsa/destevez/dennis/Lorenz/0.4_8_0.0001_0.95_0.8 $data
cp -r /data/tsa/destevez/dennis/Lorenz/0.4_6_1e-05_1.03_0.6 $data
echo "end of copy"



########## RUN ##########

cd $ESN
echo "runing............"
srun python3 tmain.py best-params -p $data -o $output -m 10 -t 0.01
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
