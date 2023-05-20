#!/bin/bash

########## EXECUTION TIME ##########

#SBATCH --time=48:00:00

########## END ##########


########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --partition=medium

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

########## END ##########





########## JOB NAME ##########

#SBATCH --job-name="Lorenz_main"

########## END ##########




########## MODULES ##########

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

########## END ##########




########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# save path
save="/data/tsa/destevez/dennis/Lorenz/main_$SLURM_JOB_ID"
mkdir -p $save

########## END ##########





########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/scripts/lorenz_srun_cpu.sh $scratch
echo "end of copy"

########## END ##########





########## RUN ##########

cd $scratch

echo ""
echo "runing............"
for i in {1..10}
do
    echo "runing $i............"
    sbatch lorenz_srun_cpu.sh $i
    echo "end of run $i"
done

echo "end of run"
echo ""

sleep 1m

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
