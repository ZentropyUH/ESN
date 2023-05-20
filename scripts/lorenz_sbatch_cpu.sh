#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000M

#SBATCH --time=48:00:00
#SBATCH --partition=medium



########## JOB NAME ##########

#SBATCH --job-name="Lorenz_main"



########## MODULES ##########

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# save path
save="/data/tsa/destevez/dennis/Lorenz/main_$SLURM_JOB_ID"
mkdir -p $save



########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/scripts/lorenz_srun_cpu.sh $scratch
echo "end of copy"



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

sleep 14400



########## SAVE ##########

echo "saving............"
cp -r $scratch $save
echo "end of save"



########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
