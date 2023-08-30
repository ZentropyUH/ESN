# ESN
An Echo State Network implementation integrated with a general scheme to use broad range of other reservoirs.


### Slurm

To start the grid search in Slurm, first create a folder with the system name, then a subfolder with the system parameters if it is necessary.

Now run the command `grid-init` from `main.py` to initialize all necessary folders and files.

E.g. `python3 main.py grid-init --path /data/tsa/destevez/MG/17 --job-name MG17 --data-path /data/tsa/destevez/data/MG/17 --steps 5`

Then go to `info/0` and execute `sbatch script.sh`. The results will be stored in `run_0/data`

To generate the next step of the grid search run the command `grid-aux` from `main.py` to find the best result, generate the new hyperparameters combinations and generate all the necesary folders and files.

E.g. `python3 main.py grid-aux --job-name MG17_2 --run-path /data/tsa/destevez/MG/17/run_0 --data-path /data/tsa/destevez/data/MG/17 --info-path /data/tsa/destevez/MG/17/info --n-results 2 --threshold 0.001 --steps 5`, and now go to `info/1` and execute `sbatch script.sh`, and do the same for the rest of runs.

### Folders
< System name >
---- < Systems params >
---- ---- info
---- ---- ---- < x >
---- ---- run_< x >
---- ---- ---- data
---- ---- ---- results

In data and results:

< name >
---- trained_model
---- forecast
---- forecast_plots
---- time.txt
---- rmse
---- rmse_mean


