#!/bin/bash

# TODO: max time
# Expected maximum runtime of job
#SBATCH --time=48:00:00

# Number of processor cores (i.e. tasks)
#SBATCH --ntasks=1
#SBATCH --partition=graphic

# We will use 4 A100 GPUs
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=2 # every task gets 2 CPUs
#SBATCH --mem-per-cpu=125000M

# Get notification to mail
#SBATCH --mail-type END


# TODO: job name
#SBATCH --job-name="KS_dennis"


## SLURM_TASK_ID  (Variable para paralelizar)

set -e

# load modules
module purge
module load cuda/11.4 # adjust cuda version as needed and add all other modules you used for development
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch


# CHECK: outputs
output="$scratch/out/"
mkdir -p $output

#SBATCH --output=$output/
#SBATCH --error=$output/



# Copy your program (and maybe input files if you need them)
ESN="$scratch/ESN/"
mkdir -p $ESN
cp -r /data/tsa/destevez/ESN/ $ESN

# TODO: data path
data="$ESN/systems/data/KS/35/N64/"

srun python3 $ESN/grid_script.py -o $output -d $data -n 5 -m 2


# TODO: output directory
# copy results to an accessable location
# only copy things you really need
save="/data/tsa/destevez/dennis/"
mkdir -p $save
cp -r $output $save


# Clean up after yourself
rm -rf $scratch

# exit gracefully
exit 0