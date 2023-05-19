#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=1:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=medium

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

########## END ##########





########## JOB NAME ##########

<<<<<<<< HEAD:scripts/KS_batch.sh
#SBATCH --job-name="KS_batch"
========
#SBATCH --job-name="test"
>>>>>>>> c05fa4d497c22da628ea7734867ee62b16b7f5c2:scripts/test.sh

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

# Script output
<<<<<<<< HEAD:scripts/KS_batch.sh
# output="$scratch/output"
# mkdir -p $output

# save path
save="/data/tsa/destevez/dennis/batch_$SLURM_JOB_ID"
========
output="$scratch/output.out"
mkdir -p $output

# save path
save="/data/tsa/destevez/dennis/test_$SLURM_JOB_ID"
>>>>>>>> c05fa4d497c22da628ea7734867ee62b16b7f5c2:scripts/test.sh
mkdir -p $save

########## END ##########



########## COPY ##########

# Copy project files to scratch
echo "copying project............"
<<<<<<<< HEAD:scripts/KS_batch.sh
cp -r /data/tsa/destevez/dennis/ESN/scripts/test.sh $scratch
========
cp -r /data/tsa/destevez/dennis/ESN/test/test.py $scratch
>>>>>>>> c05fa4d497c22da628ea7734867ee62b16b7f5c2:scripts/test.sh
echo "end of copy"

########## END ##########





########## RUN ##########

cd $ESN
echo "runing............"
<<<<<<<< HEAD:scripts/KS_batch.sh
sbatch $scratch/test.sh
========
srun python3 $scratch/test.py -p $output > $output/output.out
>>>>>>>> c05fa4d497c22da628ea7734867ee62b16b7f5c2:scripts/test.sh
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
