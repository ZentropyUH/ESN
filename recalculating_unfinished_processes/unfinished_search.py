import os
import json
from pathlib import Path


def generate_script(
        job_name: str,
        array: list,
        combinations_path: str,
        output_path: str,
        data_path: str,
        file_path:str
):
    '''
    Generate new slurm script.
    '''
    combinations_path: Path = Path(combinations_path)

    combinations_file = combinations_path.absolute().name
    output_path: Path = Path(output_path)
    data_path: Path = Path(data_path)

    file = f'''#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="{job_name}"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2000M



#SBATCH --time=02:00:00
#SBATCH --partition=graphic

#SBATCH --array={', '.join(array)}


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
cp -r {data_path.absolute()}/* $data
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
    with open(file_path, 'w') as f:
        f.write(file)


def search_unfinished(path, deep=0):
    comb_path = os.path.join(path, f'info_{deep}', 'combinations.json')
    # print('comb_path',comb_path)
    folder = os.path.join(path, f'run_{deep}')
    # print('folder',folder)
    combinations = {}
    if os.path.exists(comb_path):
        with open(comb_path, 'r') as f:
            combinations = json.load(f)
    else:
        print(f'not exist {comb_path}')
        return

    unfinished = []

    for i in combinations.keys():
        if i not in os.listdir(folder):
            unfinished.append(i)
            # print(f"NO esta la carpeta{i}")
        else:
            if  'time.txt' not in os.listdir(os.path.join(folder, i)):
                # print(os.path.join(folder, i))
                unfinished.append(i)  # [i] = combinations[i]
                os.rmdir(os.path.join(folder, i))

                # print(f"la carpeta{i} no tiene time.txt")

    if len(unfinished) == 0:
        print("All combinations terminated")
        return
    # print(unfinished)

    # save the dict in a json
    # with open(os.path.join(folder, 'unfinished.json'), 'w') as f:
    #     json.dump(unfinished, f, indent=4, sort_keys=True, separators=(",", ": "))

    generate_script(
        job_name="Recalculating unfinished processes",
        array=unfinished,
        output_path=folder,
        data_path=path,
        combinations_path=comb_path,
        file_path=os.path.join(path, 'unfinished.sh')
    )


folds=['/data/tsa/destevez/thesis/MG/16.8','/data/tsa/destevez/thesis/MG/17.0','/data/tsa/destevez/thesis/MG/30.0']
# fold = 'destevez@193.175.8.13:/data/tsa/destevez/thesis/MG/16.8'
# fold='destevez@193.175.8.13:/data/tsa/destevez/thesis/MG/17'
# fold='destevez@193.175.8.13:/data/tsa/destevez/thesis/MG/30'


# fold = '/home/lauren/Documentos/pruebas'
# search_unfinished(fold, deep=0)


for fold in folds:
    search_unfinished(fold, deep=0)
