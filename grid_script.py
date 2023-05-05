from grid_tools import *
import shutil
import time

'''
<output dir>
---- results
---- output
---- ---- <name>
---- ---- ---- trained_model
---- ---- ---- predictions
---- ---- ---- mse
---- ---- ---- mse_mean
'''      


def grid(combinations:list[list], data:list[str], output_path:str, queue_size:int, u:int=5000, tl:int=20000, threshold:float=0.01, train_time:list=[], forecast_time:list=[]):

    # Queue for best cases, n is the max number of cases
    best = Queue(queue_size)

    #Create all the combinations of hyperparameters
    for combination in product(*combinations):
        
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


        # List of Threads
        forecast_list: list[Thread] = []
        
        for fn, current_data in enumerate(data):
            
            # Thread for forecast
            current = Thread(
                target = forecast,
                kwargs={
                    "prediction_steps": 1000,
                    "train_transient": tl,
                    "trained_model_path": trained_model_path,
                    "prediction_path": forecast_path,
                    "data_file": current_data,
                    "forecast_name": fn,
                    "trained": current_data == train_data_path,
                }
            )
            # Add Thread to queue
            forecast_list.append(current)
            
        
        # Start Threads
        start_forecast_time = time.time()
        for thread in forecast_list:
            thread.start()

        # Wait for all Threads to finish
        for thread in forecast_list:
            thread.join()
        
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
        best.decide(mean, combination, current_path, threshold)

    return best.queue


hyperparameters_to_adjust={"sigma":(0,5,0.2,lambda x,y: x+y),
                        "degree_k":(2,4,2,lambda x,y: x+y),
                        "ritch_regularization":(10e-4,10,4,lambda x,y: x*y),
                        "spectral_radio": (0.9, 10 ,0.02, lambda x,y,i: x+y*i)}
                  


def grid_search(hyperparameters_to_adjust: dict, data_path: str, output_path: str, depth: int, queue_size: int, u: int=5000, tl: int=1000, threshold: int=0.01):
    
    # List all the files on the data folder
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    # Create the output folder
    output_path = join(output_path, 'output')
    makedirs(output_path, exist_ok=True)

    results_path = join(output_path, 'results')
    makedirs(results_path, exist_ok=True)

    time_file = join(results_path, 'time.txt')
    train_time = []
    forecast_time = []
    
    #Create a list[list] with the values of every hyperparameter
    combinations = generate_combinations(hyperparameters_to_adjust)

    # First search
    first_results = grid(combinations,
        data=data,        
        output_path = output_path,
        queue_size= queue_size,
        u=u,
        tl=tl,
        threshold=threshold,
        train_time=train_time,
        forecast_time=forecast_time
    )

    # Get best results
    # data: (<mse index>, (<hyperparameters>, <path>))
    best = [(1, elem) for elem in first_results]
    
    # Generate 
    steps = {}
    steps[0]= {
        "sigma": hyperparameters_to_adjust["sigma"][2],
        "degree": hyperparameters_to_adjust["degree"][2],
        "ritch_regularization": hyperparameters_to_adjust["ritch_regularization"][2],
        "spectral_radio": hyperparameters_to_adjust["spectral_radio"][2],
        "reconection_prob": hyperparameters_to_adjust["reconection_prob"][2]
    }


    all_the_best = Queue(queue_size)
    while True:
        if not len(best):
            break
        
        iteration, best_data = best.pop(0)
        
        all_the_best.add(*best_data)

        if iteration >= depth:
            continue
        
        if not steps.get(iteration):
            steps[iteration]={
                "sigma": (steps[iteration-1]["sigma"] / (hyperparameters_to_adjust["sigma"][1] + 1)),
                "degree": (steps[iteration-1]["degree"] / (hyperparameters_to_adjust["degree"][1] + 1)),
                "ritch_regularization": (steps[iteration-1]["ritch_regularization"] / (hyperparameters_to_adjust["ritch_regularization"][1] + 1)),
                "spectral_radio": (steps[iteration-1]["spectral_radio"] / (hyperparameters_to_adjust["spectral_radio"][1] + 1)),
                "reconection_prob": (steps[iteration-1]["reconection_prob"] / (hyperparameters_to_adjust["reconection_prob"][1] + 1))
            }

        
        
        params = {
            "sigma": get_param_tuple(best_data[1][0][0], hyperparameters_to_adjust["sigma"], steps[iteration]["sigma"]),
            "degree":get_param_tuple(best_data[1][0][1], hyperparameters_to_adjust["degree"], steps[iteration]["degree"]),
            "ritch_regularization": get_ritch_param_tuple(best_data[1][0][2], hyperparameters_to_adjust["ritch_regularization"], steps[iteration]["ritch_regularization"]),
            "spectral_radio": get_param_tuple(best_data[1][0][3], hyperparameters_to_adjust["spectral_radio"], steps[iteration]["spectral_radio"]),
            "reconection_prob": get_param_tuple(best_data[1][0][4], hyperparameters_to_adjust["reconection_prob"], steps[iteration]["reconection_prob"])
        }
        
        for key in params.keys():
            initial_value, number_of_values, increment, function_of_increment = params[key]
            if initial_value < 0 :
                initial_value = 0
                params[key] = (initial_value, number_of_values, increment, function_of_increment)

        first_results = grid(generate_combinations(params),
                            data=data,         
                            output_path = output_path,
                            queue_size= queue_size,
                            u=u,
                            tl=tl,
                            threshold=threshold,
                            train_time=train_time,
                            forecast_time=forecast_time
                        )
        
        best += [(iteration + 1, elem) for elem in first_results]

    calculate_aprox_time(train_time, time_file, 'Train Time')
    calculate_aprox_time(forecast_time, time_file, 'Forecaast Time')

    for i in all_the_best.queue:
        folder = i[1][1]
        folder_name = split(folder)[1]
        shutil.copytree(folder, join(results_path, folder_name), dirs_exist_ok=True)


# The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
# The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
hyperparameters_to_adjust = {
    "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
    "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
    "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: x * y**i),
    "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
    "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
}

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparametes")
    parser.add_argument('-o', '--output', help="Output path", type=str, required=True)
    parser.add_argument('-d', '--data', help="Data path", type=str, required=True)
    parser.add_argument('-n', help="Search Tree depth", type=int, default=5)
    parser.add_argument('-m', help="Number of best to keep", type=int, default=2)
    args = parser.parse_args()


    grid_search(hyperparameters_to_adjust, args.d, args.o, args.n, args.m)
