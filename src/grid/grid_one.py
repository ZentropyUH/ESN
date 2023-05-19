from src.grid.grid_tools import *
import shutil
import time


def grid_one(combination:list, data:list[str], output_path:str, queue_size:int, u:int=5000, tl:int=20000, threshold:float=0.01, train_time:list=[], forecast_time:list=[]):

        # Select the data to train
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

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


        # Train
        start_train_time = time.time()
        train(combination, train_data_path, current_path, u, tl, 'trained_model')
        train_time.append(time.time() - start_train_time)

        start_forecast_time = time.time()
        for fn, current_data in enumerate(data):

            forecast(
                prediction_steps = 1000,
                train_transient= tl,
                trained_model_path= trained_model_path,
                prediction_path= forecast_path,
                data_file= current_data,
                forecast_name= fn,
                trained= current_data == train_data_path,
            )
        
        forecast_time.append((time.time() - start_forecast_time)/len(data))

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
