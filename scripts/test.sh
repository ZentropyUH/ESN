#!/bin/bash

# expected maximum runtime of job
#SBATCH --time=01:00:00

# number of processor cores (i.e. tasks)
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=125000M

# get notification to mail
#SBATCH --mail-type END

#SBATCH --job-name="json_test"



set -e

# load modules
module load python/3.10.5

#Create your scratch space
scratch="/scratch/destevez/ESN"
mkdir -p $scratch
cd $scratch


#SBATCH --output=/scratch/destevez/out/out.out
#SBATCH --error=/scratch/destevez/out/error.err


cp /data/tsa/destevez/ESN/test.py $scratch

mkdir -p $scratch/out

cd $scratch

srun python3 /scratch/destevez/ESN/test.py -a $scratch/out

mkdir -p /data/tsa/destevez/dennis/test
cp -r $scratch/out /data/tsa/destevez/dennis/test


# exit gracefully
exit 0
