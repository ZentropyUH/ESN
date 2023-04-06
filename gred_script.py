import os


def gred(model, hyperparameters_to_adjust:dict, spectral_ratio=1.21,p=0.5,u=6000, threshold=0.1):   
    params=[]
    for elem in hyperparameters_to_adjust.values():
        params.append([ round(elem[3](elem[0],i*elem[2]),10) for i in range(elem[1])])


   



def train(instruction:str):
    os.system(instruction)


def forecast(instruction:str):
    os.system(instruction)



hyperparameters_to_adjust={"sigma":(0,5,0.2,lambda x,y: x+y),
                        "degree_k":(2,4,2,lambda x,y: x+y),
                        "ritch_regularization":(10e-4,10,4,lambda x,y: x*y),
                        "spectral_radio": (0.9, 10 ,0.02, lambda x,y,i: x+y*i),
                        "reconection_prob": (0, 5, 0.2, lambda x,y,i: x+y*i)}

# gred(hyperparameters_to_adjust=hyperparameters_to_adjust)

params=[]
for elem in hyperparameters_to_adjust.values():
    params.append([ round(elem[3](elem[0],i*elem[2]),10) for i in range(elem[1])])

print(params)


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


