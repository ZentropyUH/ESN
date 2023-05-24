from src.grid.grid_tools import *
import shutil
import time
import tensorflow as tf
import argparse
import json


def grid_one(combination_index: int, data_path: str, output_path:str, u:int=9000, tl:int=20000):

        # Select the data to train
        data: list[str] = [join(data_path, p) for p in listdir(data_path)]
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

        # Create the output folder
        makedirs(output_path, exist_ok=True)

        # Get the combination from src/grid/combinations.json file in the index position
        combination = []
        with open('src/grid/combinations.json', 'r') as f:
            combinations = json.load(f)
            combination = combinations[str(combination_index)]
        print('Combination: {}'.format(combination))
        

        # Create the trained model folder
        current_path = join(output_path, '_'.join([str(x) for x in combination]))
        trained_model_path = join(current_path, 'trained_model')
        
        # Create forecast folder
        forecast_path = join(current_path, 'forecast')
        makedirs(forecast_path, exist_ok=True)
        
        # Create folder for the mse of predictions
        mse_path = join(current_path, 'mse')
        makedirs(mse_path, exist_ok=True)

        # Create the folder mean
        mean_path = join(current_path, 'mse_mean')
        makedirs(mean_path, exist_ok=True)

        # Create time file
        time_file = join(current_path, 'time.txt')


        # Train
        start_train_time = time.time()
        print('Training...')
        train(combination, train_data_path, current_path, u, tl, 'trained_model')
        print('Training finished')
        train_time = time.time() - start_train_time

        # Forecast
        start_forecast_time = time.time()
        for fn, current_data in enumerate(data):
            print('Forecasting {}...'.format(fn))
            forecast(
                prediction_steps = 1000,
                train_transient= tl,
                trained_model_path= trained_model_path,
                prediction_path= forecast_path,
                data_file= current_data,
                forecast_name= fn,
                trained= current_data == train_data_path,
            )
            print('Forecasting {} finished'.format(fn))
        forecast_time = (time.time() - start_forecast_time)/len(data)

        # Get Forecast data files
        forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
        
        # Calculate MSE
        mse = [[np.square(np.subtract(f, d)).mean()
                for f, d in zip(pd.read_csv(forecast_file).to_numpy(), pd.read_csv(data_file).to_numpy()[(1000 + 1000 + tl):])]
                for forecast_file, data_file in zip(forecast_data, data)]

        # Sum all the mse
        mean = []
        for i, current in enumerate(mse):
            
            # Save current mse
            save_csv(current, "{}.csv".format(i), mse_path)
            
            if len(mean) == 0:
                mean = current
            else:
                mean = np.add(mean, current)

        mean = [x / len(data) for x in mean]

        # Save the csv
        save_csv(mean, "mse_mean.csv", mean_path)
        save_plots(data=mean, output_path=mean_path, name='mse_mean_plot.png')

        with open(time_file, 'w') as f:
            json.dump({'train': train_time, 'forecast': forecast_time}, f)



def best_combinations(path: str, output: str, max_size: int, threshold: float):
    
    best = Queue(max_size)
    for folder in listdir(path):
        folder = join(path, folder)
        mse_mean_path = join(folder, 'mse_mean', 'mse_mean.csv')
        params_path = join(folder, 'params.json')

        mse_mean = []
        with open(mse_mean_path, 'r') as f:
            mse_mean = read_csv(f)
        
        params = {}
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        params = (
            params['reservoir_sigma'],
            params['reservoir_degree'],
            params['regularization'],
            params['spectral_radius'],
            params['rewiring'],
        )
        
        best.decide(mse_mean, params, folder, threshold)
    
    for i, element in enumerate(best.queue):
        folder = element[1][1]
        shutil.copytree(folder, join(output, str(i)), dirs_exist_ok=True)



def change_folders(path: str):
    
    for folder in listdir(path):
        folder = join(path, folder)
        time_file = join(folder, 'time.txt')

        inside_folders = [join(folder, f) for f in listdir(folder) if join(folder, f) != time_file]

        for inside_folder in inside_folders:
            shutil.move(time_file, inside_folder)
            shutil.move(inside_folder, path)
            shutil.rmtree(folder)
     

def detect_not_fished_jobs(path: str, output: str):
    with open(join(output, 'out.out'), 'w') as f:
        for file in [join(path, f) for f in listdir(path) if 'time.txt' not in listdir(join(path, f))]:
            f.write('{}\n'.format(file.split('_')[-1]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparametes")
    parser.add_argument('-o', '--output', help="Output path", type=str, required=True)
    parser.add_argument('-d', '--data', help="Data path", type=str, required=True)
    parser.add_argument('-i', '--index', help="Combination of hyperparameters", type=int, required=True)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print(gpus)

    grid_one(args.index, args.data, args.output)