"""Do main stuff here. Correct docstring later."""
import timeit

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


def main():
    """Tryout many things."""
    init_transient = 1000
    transient = 1000
    train = 10000

    units = 4000
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

    load_path = f"../data/KS/L{L}_dt0.25/"

    name = f"KS_L{L}_N{N}_dt0.25_steps160000_diffusion-k1_run0.csv"
    # name = "mackey_alpha-0.2_beta-10_gamma-0.1_tau-17_n-150000.csv"
    # name = "Lorenz_[2, 2, 2]_tend3200_dt0.02.csv"

    title = f"Forecasting of the Kuramoto-Sivashinsky model with {units} units"  # KS
    # title = f"Forecasting of the Mackey-Glass model with {units} units"  # Mackey-Glass
    # title = f"Forecasting of the Lorenz model with {units} units"  # Lorenz

    save_name = (
        f"{name[:-4]}_units{units}_train{train}_inittransient{init_transient}"
        f"_transient{transient}_degree{degree}_sigma{sigma}_spectral_radius{spectral_radius}"
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
        load_path + name,
        transient=transient,
        train_length=train,
        init_transient=init_transient,
    )

    # ylabels = ["X", "Y", "z"]  # For the linear plot of lorenz
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
    exit(0)

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
    exit(0)

    # plt.plot(monitored["rms_error"])
    # plt.show()

    # plot_linear_forecast(
    #     predictions,
    #     val_target,
    #     dt=dt,
    #     title=title,
    #     xlabel="t",
    #     ylabels=ylabels,
    #     save_path="plots/" + save_name,
    # )

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


if __name__ == "__main__":
    print(timeit.timeit(main, number=1))
