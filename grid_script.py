from grid_tools import *


'''
<output dir>
---- output
---- ---- <name>
---- ---- ---- trained_model
---- ---- ---- predictions
---- ---- ---- mse_mean
'''      


def grid(combinations:list[list], data:list[str], data_path:str, output_path:str, queue_size:int, u:int=5000, tl:int=1000, threshold:float=0.01):
    
    # Queue for best cases, n is the max number of cases
    best = Queue(queue_size)

    #Create all the combinations of hyperparameters
    for combination in product(*combinations):
        
        # Select the data to train
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

        # Create the output folders
        current_path = join(output_path, '_'.join([str(x) for x in combination]))
        trained_model_path = join(current_path, 'trained_model')
        forecast_path = join(current_path, 'forecast')
        makedirs(forecast_path, exist_ok=True)

        # Train
        train(combination, train_data_path, current_path, u, tl, 'trained_model')

        # List of Threads
        forecast_list: list[Thread] = []
        
        for fn, current_data in enumerate(data[:3]): # Delete [:3]
            
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
            # Start Thread
            current.start()
        
        # Wait for all Threads to finish
        for thread in forecast_list:
            thread.join()
        


        forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
        
        mse = [[np.square(np.subtract(f, d)).mean()
                for f, d in zip(pd.read_csv(forecast_file).to_numpy(), pd.read_csv(data_file).to_numpy()[(1000 + 1000 + tl):])]
                for forecast_file, data_file in zip(forecast_data, data)]

        # Sum all the mse
        mean = []
        for current in mse:
            if mean == []:
                mean = current
            else:
                mean = np.add(mean, current)
                # list(map(lambda x, y: x+y, mean, current))

        mean = [x / len(data[:3]) for x in mean] # Delete [:3]
        
        # Create the folder mean
        mean_path = join(current_path, 'mse_mean')
        makedirs(mean_path, exist_ok=True)

        # Save the csv
        save_csv(mean, "mse_mean.csv", mean_path)

        best.decide(mean, combination, threshold)

        break # Fucking delete later

    return best.queue




def get_param_tuple(value, param , step):    
    initial_value, number_of_values, increment, function_of_increment = param
    initial_value=value-int(number_of_values/2)*step
    return initial_value,number_of_values,step,function_of_increment

def grid_search(hyperparameters_to_adjust:dict, data_path, output_path, depth:int, queue_size:int, u=5000, tl=1000, threshold=0.01):
    
    # List all the files on the data folder
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    # Create the output folder
    output_path = join(output_path, 'output')
    makedirs(output_path, exist_ok=True)
    
    #Create a list[list] with the values of every hyperparameter
    combinations = generate_combinations(hyperparameters_to_adjust)


    best_results = grid(combinations,
        data=data, 
        data_path = data_path,         
        output_path = output_path,
        queue_size= queue_size,
    )

    better = [(1, elem) for  elem in best_results]
    steps={}

    steps[0]= {
        "sigma":hyperparameters_to_adjust["sigma"][2],
        "degree":hyperparameters_to_adjust["degree"][2],
        "ritch_regularization":hyperparameters_to_adjust["ritch_regularization"][2],
        "spectral_radio":hyperparameters_to_adjust["spectral_radio"][2],
        "reconection_prob":hyperparameters_to_adjust["reconection_prob"][2]
    }

    all_the_best=[]
    while True:
        if not len(better):
            break
        
        iteration, combination = better.pop(0)
        
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

        all_the_best.append(combination)
        
        params = {
            "sigma": get_param_tuple(combination[1][0], hyperparameters_to_adjust["sigma"], steps[iteration]["sigma"]),
            "degree":get_param_tuple(combination[1][1], hyperparameters_to_adjust["degree"], steps[iteration]["degree"]),
            "ritch_regularization": get_param_tuple(combination[1][2], hyperparameters_to_adjust["ritch_regularization"], steps[iteration]["ritch_regularization"]),
            "spectral_radio": get_param_tuple(combination[1][3], hyperparameters_to_adjust["spectral_radio"], steps[iteration]["spectral_radio"]),
            "reconection_prob": get_param_tuple(combination[1][4], hyperparameters_to_adjust["reconection_prob"], steps[iteration]["reconection_prob"])
        }
        
        best_results = grid(generate_combinations(params),
                            data=data, 
                            data_path = data_path,         
                            output_path = output_path,
                            queue_size= queue_size
                        )
        
        better.append((iteration+1,elem) for  elem in best_results)

    return all_the_best




# The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
# The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
hyperparameters_to_adjust = {
    "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
    "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
    "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: round(x * y**i, 8)),
    "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
    "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
}


grid_search(
    hyperparameters_to_adjust,
    '/media/dionisio35/Windows/_folders/_new/22/',
    '/media/dionisio35/Windows/_folders/_new/',
    5,
    5,
)


# grid_search(hyperparameters_to_adjust, 
#             data_path="/home/lauren/Documentos/ESN/data/MG",         
#             output_path="/home/lauren/Documentos/ESN/forecasting",
#             depth=5,
#             queue_size = 3) 

