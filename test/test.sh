#!/bin/bash
#SBATCH --job-name=myarrayjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=5000M
#SBATCH --time=4:00:00
#SBATCH --partition=graphic



scratch="/scratch/$USER/"

ls $scratch