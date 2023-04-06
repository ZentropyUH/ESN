import os


def gred(model, hyperparameters_to_adjust:dict, sigma:list, degree_k:list, ritch_regularization:list,spectral_ratio=1.21,p=0.5,u=6000, ):   
    
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





def train(instruction:str):
    os.system(instruction)


def forecast(instruction:str):
    os.system(instruction)

    # os.system("python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm '/home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb' -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000")
    # os.system("python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/keras_metadata.pb -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000")
    # python3 ./main.py forecast -fm classic -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000' -o /home/lauren/Documentos/ESN/forecasting -df '/home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv' -it 1000 -tr 10000 -tl 10000

# "/home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'"
# os.system("python3 ./main.py forecast -fm classic  -fl 1000 -sil 50 -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'-rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb -it 1000 -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv -tr 1000 -tl 10000  -o /home/lauren/Documentos/ESN/forecasting")





hyperparameters_to_adjust

# gred(sigma=[0.2,0.4,0.6,0.8,1],degree_k=[2,4,6,8],ritch_regularization=[10e-4,10e-5,10e-6,10e-7,10e-8])







    # train("python3 ./main.py train \
    #         -m ESN \
    #         -ri WattsStrogatzOwn\
    #         -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv \
    #          -o'/home/lauren/Documentos/ESN/trained'  \
    #          -rs 0.2 \
    #          -sr 1.21 \
    #         -rw 0.5 \
    #         -u 6000 \
    #         -rd 2 \
    #         -rg 10e-4")




os.system("python3 ./main.py forecast \
        -fm classic \
        -fl 1000 \
        -sil 50\
        -tm /home/lauren/Documentos/ESN/trained/dta_'MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv'-inp_scl_'0.5'-lr_'1.0'-mdl_'ESN'\
        -rdout_'linear'-reg_'0.001'-res_deg_'2'-res_std_'0.2'-rw_'0.5'-sp_rad_'1.21'-train_len_'10000'-units_'6000'/saved_model.pb -it 1000 \
        -df /home/lauren/Documentos/ESN/data/MG/16.8/MG_tau16.8_dt0.05_n250000_t-end12500.0_seed2636.csv -tr 1000 -tl 10000 \
        -o /home/lauren/Documentos/ESN/forecasting")


