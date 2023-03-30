"""Do main stuff here. Correct docstring later."""
import timeit
from os.path import realpath, join, dirname

from keras.models import load_model

import numpy as np

# from scipy.signal import (  # This is to have relative maximums and minimums.
#     argrelmax,
#     argrelmin,
# )
from tensorflow import keras

from custom_initializers import *
from custom_models import *
from custom_layers import *
from readout_generators import *
from utils import *


from model_instantiators import *
from forecasters import *
from plotters import *

# To avoid tensorflow verbosity
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# To use the CPU instead of the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""



from os.path import realpath, join, dirname

def generation(data_path, units, train, init_transient, transient, degree, sigma,
               spectral_radius, regularization, rewiring_p):

    (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data(
        data_path,
        transient=transient,
        train_length=train,
        init_transient=init_transient,
    )
    
    reservoir_init = WattsStrogatzOwn(
        degree=degree,
        spectral_radius=spectral_radius,
        sigma=sigma,
        rewiring_p=rewiring_p,
    )
    
    model:ESN = get_simple_esn(
        units=units,
        # seed=seed,
        spectral_radius=spectral_radius,
        degree=degree,
        sigma=sigma,
        reservoir_initializer=reservoir_init,
        leak_rate=1,
    )
    
    model.verify_esp(transient_data[:, :50, :], times=10)

    final_model = linear_readout(
        model,
        transient_data,
        train_data,
        train_target,
        regularization=regularization,
        method="ridge",
    )

    print("Training finished, number of parameters: ", final_model.count_params())

    final_model.save("model")

    keras.utils.plot_model(
        final_model.build_graph(),
        # show_shapes=True,
        to_file="model.png",
    )


def predict(data_path, output_name, L, N, init_transient, transient, train, forecast_len):
    (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data(
        data_path,
        transient=transient,
        train_length=train,
        init_transient=init_transient,
    )


    final_model = keras.models.load_model("model")
    # final_model = load_model("model")

    predictions = classic_forecast(
        final_model,
        forecast_transient_data,
        val_data,
        val_target,
        forecast_length=forecast_len,
        save_name= output_name
    )

    ylabels = ["X", "Y", "z"]
    yvalues = np.linspace(0, L, train_data.shape[-1])

    plot(L, predictions, val_target, yvalues, ylabels)


def plot(L, predictions, val_target, yvalues, ylabels):
    max_lyap = lyap_ks(1, L)
    dt = 0.25 * max_lyap
    print("Max Lyapunov exponent: ", max_lyap)

    # plot_linear_forecast(
    #     predictions,
    #     val_target,
    #     dt=dt,
    #     # title=title,
    #     xlabel="t",
    #     # ylabels=ylabels,
    #     save_path="D:\\data\\" + 'a',
    # )
    
    plot_contourf_forecast(
        predictions,
        val_target,
        dt=dt,
        # title=title,
        save_path="plots/" + 'a',
        show=True,
        # yvalues=yvalues,
    )

    # try:
    #     plot_linear_forecast(
    #         predictions,
    #         val_target,
    #         dt=dt,
    #         # title=title,
    #         xlabel="t",
    #         ylabels=ylabels,
    #         save_path="D:\\data\\" + 'a',
    #     )
    #     print('successful plot_linear_forecast')
    # except Exception as e:
    #     print('failed plot_linear_forecast')
    #     print(e)
    

    # try:
    #     plot_contourf_forecast(
    #         predictions,
    #         val_target,
    #         dt=dt,
    #         # title=title,
    #         save_path="plots/" + 'a',
    #         show=True,
    #         yvalues=yvalues,
    #     )
    #     print('successful plot_contourf_forecast')
    # except Exception as e:
    #     print('failed plot_contourf_forecast')
    #     print(e)



def stuff():
    init_transient = 1000
    transient = 1000
    train = 20000
    units = 6000
    forecast_len = 350

    # variate
    spectral_radius = 0.9
    degree = 4
    sigma = 0.5
    regularization = 10e-4
    rewiring_p= 0.5

    seed= 1000

    #for multi-reservoir
    reservoir_amount = 8
    overlap = 3

    L = 22
    N = 128

    path= realpath(f"D:\data\KS\L{L}_N{N}_dt0.25")
    name = f"KS_L{L}_N{N}_dt0.25_steps160000_diffusion-k1_run0.csv"

    title = f"Forecasting of the Kuramoto-Sivashinsky model with {units} units"

    generation(join(path, name), units, train, init_transient, transient, degree, sigma,
               spectral_radius, regularization, rewiring_p)
    

    for n in range(0, 30):
        name = f"KS_L{L}_N{N}_dt0.25_steps160000_diffusion-k1_run{n}.csv"
        output_name= f'D:\\data\\out\\output{n}'
        predict(join(path, name), output_name, L, N, init_transient, transient, train, forecast_len)
    


stuff()


def main():
    """Tryout many things."""
    init_transient = 1000
    transient = 1000
    train = 20000

    units = 6000
    spectral_radius = 0.45  # KS
    # spectral_radius = 1.21 # Mackey-Glass

    reservoir_amount = 8
    overlap = 3

    degree = 3  # KS
    # degree = 2  # Mackey-Glass
    # degree = 6  # Lorenz

    # sigma = 1
    sigma = 0.5  # KS
    # sigma = 0.1  # Lorenz

    forecast_len = 350

    regularization = 1e-4  # KS
    # regularization = 1e-8  # Mackey-Glass/Lorenz

    L = 22  # KS
    N = 128  # KS

    max_lyap = lyap_ks(1, L)
    dt = 0.25 * max_lyap  # KS # delta t and max lyapunov time

    print("Max Lyapunov exponent: ", max_lyap)

    load_path = realpath(f"D:\data\KS\L{L}_N{N}_dt0.25")

    name = f"KS_L{L}_N{N}_dt0.25_steps160000_diffusion-k1_run0.csv"
    # name = "mackey_alpha-0.2_beta-10_gamma-0.1_tau-17_n-150000.csv"
    # name = "Lorenz_[2, 2, 2]_tend3200_dt0.02.csv"

    title = f"Forecasting of the Kuramoto-Sivashinsky model with {units} units"  # KS
    # title = f"Forecasting of the Mackey-Glass model with {units} units"  # Mackey-Glass
    # title = f"Forecasting of the Lorenz model with {units} units"  # Lorenz

    save_name = join(
        f"{name[:-4]}_units{units}_train{train}_inittransient{init_transient}",
        f"_transient{transient}_degree{degree}_sigma{sigma}_spectral_radius{spectral_radius}",
        f"_regularization{regularization}_forecast_len{forecast_len}"
    )

    (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data(
        join(load_path, name),
        transient=transient,
        train_length=train,
        init_transient=init_transient,
    )

    ylabels = ["X", "Y", "z"]  # For the linear plot of lorenz
    yvalues = np.linspace(0, L, train_data.shape[-1])

    input_init = None
    bias_init = None

    # input_init = keras.initializers.RandomUniform(minval=-sigma, maxval=sigma)
    # bias_init = keras.initializers.RandomUniform(minval=-sigma, maxval=sigma)
    # reservoir_init = WattsStrogatzOwn(
    #     degree=degree,
    #     spectral_radius=spectral_radius,
    #     sigma=sigma,
    #     rewiring_p=0.5,
    # )

    # reservoir_init = RegularNX(
    #     degree=degree, spectral_radius=spectral_radius, sigma=sigma
    # )

    # reservoir_init = WattsStrogatzOwn(
    #     degree=degree, spectral_radius=spectral_radius, sigma=sigma
    # )

    # Simple ESN
    model = get_simple_esn(
        units=units,
        #   seed=seed,
        spectral_radius=spectral_radius,
        degree=degree,
        sigma=sigma,
        # input_initializer=input_init,
        # bias_initializer=bias_init,
        # reservoir_initializer=reservoir_init,
        leak_rate=1,
    )

    # # Parallel ESN
    # model = get_parallel_esn(
    #     units_per_reservoir=units,
    #     reservoir_amount=reservoir_amount,
    #     overlap=overlap,
    #     #   seed=seed,
    #     spectral_radius=spectral_radius,
    #     degree=degree,
    #     sigma=sigma,
    #     input_initializer=input_init,
    #     bias_initializer=bias_init,
    #     reservoir_initializer=reservoir_init,
    # )

    # model.verify_esp(transient_data[:, :50, :], times=10)
    # exit(0)

    final_model = linear_readout(
        model,
        transient_data,
        train_data,
        train_target,
        regularization=regularization,
        method="ridge",
    )

    # final_model = sgd_linear_readout(
    #     model,
    #     transient_data,
    #     train_data,
    #     train_target,
    #     learning_rate=0.001,
    #     epochs=30,
    #     regularization=regularization,
    # )

    print(
        "Training finished, number of parameters: ", final_model.count_params()
    )

    keras.utils.plot_model(
        final_model.build_graph(),
        # show_shapes=True,
        to_file="model.png",
    )

    # predictions = classic_forecast(
    #     final_model,
    #     forecast_transient_data,
    #     val_data,
    #     val_target,
    #     forecast_length=forecast_len,
    # )

    final_model.save("caca")

    print("saved the fucking model")

    final_model = keras.models.load_model(
        "caca",
        # custom_objects=custom_objects,
    )
    print("Model loaded")

    predictions = section_forecast(
        final_model,
        forecast_transient_data,
        val_data,
        val_target,
        section_initialization_length=50,
        section_length=100,
        number_of_sections=10,
    )

    print(predictions.shape)

    # plt.plot(monitored["rms_error"])
    # plt.show()

    plot_linear_forecast(
        predictions,
        val_target,
        dt=dt,
        title=title,
        xlabel="t",
        # ylabels=ylabels,
        save_path="plots/" + save_name,
    )

    plot_contourf_forecast(
        predictions,
        val_target,
        dt=dt,
        title=title,
        save_path="plots/" + save_name,
        show=True,
        yvalues=yvalues,
    )

    print(val_target.shape)
    print(predictions.shape)

    # render_video(
    #     data=val_target,
    #     predictions=predictions,
    #     title=title,
    #     frames=forecast_len,
    #     save_path="videos/" + save_name + "_video" + ".mp4",
    #     dt=dt,
    # )


# # # # # # For hyperas


# if __name__ == "__main__":
#     print(timeit.timeit(main, number=1))
