# ESN
An Echo State Network implementation integrated with a general scheme to use broad range of other reservoirs.


### Slurm

To start the grid search in Slurm, first create a folder with the system name, then a subfolder with the system parameters if it is necessary.

Now run the command `init-slurm-grid` from `main.py` to initialize all necessary folders and files. When you use the command you must set the output path created above in `--path`, if it dosent exist it will be generated any way, then `--job-name` for the slurm job name and --data-path for the dataset path. Then just add all the parameters. To use various parameters of the same type just repeat it like the example below.

E.g. `python3 main.py init-slurm-grid --path /data/tsa/destevez/dennis/MG/16.8/ --job-name MG16.8 --data-path /data/tsa/destevez/data/MG/16.8/ --units 3000 --train-length 20000 --forecast-length 1000 --transient 1000 --steps 5 --input-scaling 0.5 -lr 0 -lr 0.2 -lr 0.4 -lr 0.6 -lr 0.8 -lr 1.0 -sr 0.9 -sr 0.95 -sr 1 -sr 1.05 -rw 0.05 -rw 0.1 -rw 0.15 -rw 0.2 -rd 2 -rd 4 -rd 6 -rd 8 -rs 0.2 -rs 0.4 -rs 0.6 -rs 0.8 -rs 1.0 -rg 1e-4 -rg 1e-5 -rg 1e-6 -rg 1e-7 -rg 1e-8`

Then go to `info_0` and execute `sbatch script.sh`. The results will be stored in `run_0/data`.

### Folders
< System name >
---- < Systems params >
---- ---- info_< x >
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


