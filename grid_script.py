from os import makedirs, listdir, system
from os.path import join, isdir

from threading import Thread
from random import randint
from itertools import product


'''
<output dir>
---- output
---- ---- <name>
---- ---- ---- trained_model
---- ---- ---- predictions

'''

# TODO: Paralelizar el proceso de ejecucion de todos los casos con mismo set de hiperparametros
# TODO: Entrenar un caso random y guardar su modelo
# TODO: Luego de tener las predicciones del set de parametros se halla la media entre todas
# TODO: Se lleva una lista con los n mejores casos
# TODO: Se elimina el caso que deja de pertenecer a la lista
# TODO: Modelar para permitir diferentes hiperparametros a ajustar


def grid(hyperparameters_to_adjust:dict, data_path, output_path, u=1000, threshold=0.1):

    # List all the files on the data folder
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    # Create the output folder
    output_path = join(output_path, 'output')
    # makedirs(output_path)

    
    #Create a list[list] with the values of every hyperparameter
    params: list[list] = [[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in hyperparameters_to_adjust.values()]
    
    #Create all the combinations of hyperparameters
    for combination in product(*params):
        
        # Select the data to train
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

        # Create the output folders
        current_path = join(output_path, ''.join([str(x) for x in combination]))
        trained_model_path = join(current_path, 'trained_model')
        makedirs(trained_model_path)
        prediction_path = join(current_path, 'predictions')
        makedirs(prediction_path)

        # Train
        train(combination, train_data_path, trained_model_path, u)

        # List of Threads
        prediction_list: list[Thread] = []
        
        for current_data in data: 
            current = Thread(
                target = forecast,
                args=(
                    1000,
                    1000,
                    u,
                    trained_model_path,
                    prediction_path,
                    current_data == train_data_path
                )
            )
            
            prediction_list.append(current)
            current.start()
        
        for thread in prediction_list:
            thread.join()
            



def train(params, data_file_path, output_file, u):    
    instruction = f"python3 ./main.py train \
            -m ESN \
            -ri WattsStrogatzOwn\
            -df {data_file_path} \
            -o {output_file} \
            -rs {params[0]} \
            -sr {params[3]} \
            -rw {params[4]} \
            -u {u} \
            -rd {params[1]} \
            -rg {params[2]}"

    system(instruction)


def forecast(prediction_steps: int, init_transient: int, train_transient: int, trained_model_path: str, prediction_path: str, trained: bool):
    if trained:
        instruction = f"python3 ./main.py forecast \
                -fm classic \
                -fl {prediction_steps} \
                -it {init_transient} \
                -tl {train_transient} \
                -tm {trained_model_path} \
                -o {prediction_path}"
    else:
        instruction = f"python3 ./main.py forecast \
                -fm classic \
                -fl {prediction_steps} \
                -it {init_transient} \
                -tm {trained_model_path} \
                -o {prediction_path}"

    system(instruction)


# los hiperparametros van a ser de la forma: nombre:(valor_inicial,numero_de_valores,incremento,funcion_de_incremento)
# los parametros de la funcion de incremento son: valor_inicial,incremento,valor_actual_de_la_iteracion
hyperparameters_to_adjust = {"sigma": (0, 5, 0.2, lambda x,y,i: x+y*i),
                        "degree": (2, 4, 2, lambda x,y,i: x+y*i),
                        "ritch_regularization": (10e-5, 5, 0.1,lambda x,y,i: x*y**i),
                        "spectral_radio": (0.9, 10 ,0.02, lambda x,y,i: x+y*i),
                        "reconection_prob": (0.2, 5, 0.2, lambda x,y,i: x+y*i)
                    }



grid(hyperparameters_to_adjust, 
        data_path = '/media/dionisio35/Windows/_new/22/',         
        output_path = '/media/dionisio35/Windows/_new/') 
