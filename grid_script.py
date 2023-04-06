import os
from itertools import product
from enum import Enum

class Parameters(Enum):
    SIGMA=0
    DEGREE=1
    REGULARIZATION=2
    SPECTRAL_RATIO=3
    RECONECTION_PROB=4




def grid(hyperparameters_to_adjust:dict, data_file_path, output_file, u=6000, threshold=0.1):   
    params=[]
    for elem in hyperparameters_to_adjust.values():# crea una lista de listas de los valores que puede tomar cada hiperparametro
        params.append([elem[3](elem[0],elem[2],i) for i in range(elem[1])])
    for combination in product(*params):# crea todas las combinaciones de los hiperparametros
        train(combination,data_file_path, output_file,u)
        break



def train(params, data_file_path,output_file,u):
    instruction=f"python3 ./main.py train \
            -m ESN \
            -ri WattsStrogatzOwn\
            -df {data_file_path} \
            -o {output_file} \
            -rs {params[Parameters[SIGMA]]} \
            -sr {params[Parameters.SPECTRAL_RATIO]} \
            -rw {params[Parameters.RECONECTION_PROB]} \
            -u {u} \
            -rd {params[Parameters.DEGREE]} \
            -rg {params[Parameters.REGULARIZATION]}"

    os.system(instruction)


def forecast(instruction:str):
    os.system(instruction)


# los hiperparametros van a ser de la forma: nombre:(valor_inicial,numero_de_valores,incremento,funcion_de_incremento)
# los parametros de la funcion de incremento son: valor_inicial,incremento,valor_actual_de_la_iteracion
hyperparameters_to_adjust = {"sigma":(0,5,0.2,lambda x,y,i: x+y*i),
                        "degree":(2,4,2,lambda x,y,i: x+y*i),
                        "ritch_regularization":(10e-5,5,0.1,lambda x,y,i: x*y**i),
                        "spectral_radio": (0.9, 10 ,0.02, lambda x,y,i: x+y*i),
                        "reconection_prob": (0, 5, 0.2, lambda x,y,i: x+y*i)}



grid(hyperparameters_to_adjust, 
        data_file_path="/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv",         
        output_file="/home/lauren/Documentos/ESN/forecasting") 




 ##Example
    # train("python3 ./main.py train \
    #         -m ESN \
    #         -ri WattsStrogatzOwn\
    #         -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv \
    #         -o /home/lauren/Documentos/ESN/trained \
    #         -rs 0.2 \
    #         -sr 1.21 \
    #         -rw 0.5 \
    #         -u 6000 \
    #         -rd 2 \
    #         -rg 10e-4")

    # forecast("python3 ./main.py forecast \
    #     -fm classic \
    #     -fl 1000 \
    #     -sil 50\
    #     -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'\
    #     -rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb -it 1000 \
    #     -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv -tr 1000 -tl 10000 \
    #     -o /home/lauren/Documentos/ESN/forecasting")



# os.system("python3 ./main.py forecast \
#         -fm classic \
#         -fl 1000 \
#         -sil 50\
#         -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'\
#         -rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb -it 1000 \
#         -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv -tr 1000 -tl 10000 \
#         -o /home/lauren/Documentos/ESN/forecasting")



    # os.system("python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm '/home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb' -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000")
    # os.system("python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/keras_metadata.pb -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000")
    # python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000' -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000

# "/home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'"
# os.system("python3 ./main.py forecast -fm classic  -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb -it 1000 -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv -tr 1000 -tl 10000  -o /home/lauren/Documentos/ESN/forecasting")


