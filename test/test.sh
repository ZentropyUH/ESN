#!/bin/bash
#SBATCH --job-name=myarrayjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10

#SBATCH --mem-per-cpu=5000M
#SBATCH --time=4:00:00
#SBATCH --partition=medium



scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

cp -r /data/tsa/destevez/dennis/ESN/test/config.txt $scratch
config=$scratch/config.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
sample=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

# Extract the sex for the current $SLURM_ARRAY_TASK_ID
sex=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

echo "1"

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task , the sample name is ${sample} and the sex is ${sex}."