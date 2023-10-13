from os.path import join
from enum import Enum


class BaseFolders(Enum):
    pass


# BASE FOLDERS
class GridFolders(BaseFolders):
    '''
    Base folders generated for the grid search.
    '''
    INFO = 'info_{depth}'
    RUN = 'run_{depth}'


# RUN FOLDERS
class RunFolders(BaseFolders):
    '''
    Folders for the
    '''
    RUN_DATA = 'data'
    RUN_RESULTS = 'results'


# INFO FILES
class InfoFiles(BaseFolders):
    INFO_FILE = 'info.json'
    COMBINATIONS_FILE = 'combinations.json'
    SLURM_FILE = 'script.sh'


# GRID FOLDERS
class CaseRun(BaseFolders):
    FORECAST = 'forecast'
    RMSE = 'rmse'
    RMSE_MEAN = 'rmse_mean'
    TRAINED_MODEL = 'trained_model'
    FORECAST_PLOTS = 'forecast_plots'
    TIME_FILE = 'time.json'
    RMSE_MEAN_FILE = join('rmse_mean', 'rmse_mean.csv')
    RMSE_MEAN_PLOT_FILE = join('rmse_mean', 'rmse_mean_plot.png')


SLURM_SCRIPT = '''#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="{job_name}"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M



#SBATCH --time=4-00:00:00
#SBATCH --partition=graphic

#SBATCH --array={array}


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
ESN=./ESN
mkdir -p $ESN

# output path
output="./output"
mkdir -p $output

# data path
data="./data"
mkdir -p $data

# save path
save="{output_path}"
mkdir -p $save

# combinations
comb=$ESN/{combinations_file}


########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying project............"
cp -r "{combinations_path}" $ESN

echo "copying data............"
cp -r {data_path}/* $data
echo "end of copy"



########## RUN ##########

echo "runing............"
srun python3 ESN/main.py slurm-grid -d $data -o $output -i $SLURM_ARRAY_TASK_ID -hp $comb
echo "end of run"



########## SAVE ##########

echo "saving............"
cp -r $output/* $save
echo "end of save"



########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
'''
