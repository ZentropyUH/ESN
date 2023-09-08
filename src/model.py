import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from typing import Any
import numpy as np
import tensorflow as tf
from rich.progress import track
from sklearn.linear_model import ElasticNet, Lasso, Ridge

import keras
import keras.layers

from src.customs.custom_layers import EsnCell, PowerIndex
from src.utils import tf_ridge_regression

# TODO: save predict in scv
# TODO: separate prediction and evaluation

class ESN:
    def __init__(self, inputs, outputs, readout) -> None:
        self.inputs: keras.layers.Layer = inputs
        self.outputs: keras.layers.Layer = outputs
        self.readout: keras.layers.Layer = readout
        self.reservoir: keras.Model = None
        self.model: keras.Model = None
    
    # TODO: Cant predict if not trained
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)

    # TODO: Cant predict if not trained
    def predict(self, inputs):
        return self.model.predict(inputs)
    
    # FIX: Exeption for different features
    def train(
        self,
        transient_data: np.ndarray,
        train_data: np.ndarray,
        train_target: np.ndarray,
        regularization: int,
    ):
        if not self.reservoir:
            self.reservoir = keras.Model(
                inputs=self.inputs,
                outputs=self.outputs
            )
        
        print("\nEnsuring ESP...\n")
        if not self.reservoir.built:
            self.reservoir.build(input_shape=transient_data.shape)
        self.reservoir.predict(transient_data)

        print("\nHarvesting...\n")
        start = time.time()
        harvested_states = self.reservoir.predict(train_data)
        end = time.time()
        print(f"Harvesting took: {round(end - start, 2)} seconds.")
        
        print("Calculating the readout matrix...\n")
        method = 'ridge'
        solver = 'svd'
        if method == "ridge":
            readout = Ridge(alpha=regularization, tol=0, solver=solver)
        elif method == "lasso":
            readout = Lasso(alpha=regularization, tol=0)
        elif method == "elastic":
            readout = ElasticNet(
                alpha=regularization, tol=1e-4, selection="random"
            )
        else:
            raise ValueError("The method must be ['ridge' | 'lasso' | 'elastic'].")

        readout.fit(harvested_states[0], train_target[0])
        predictions = readout.predict(harvested_states[0])

        training_loss = np.mean((predictions - train_target[0]) ** 2)
        print(f"Training loss: {training_loss}\n")

        self.readout.build(harvested_states[0].shape)
        self.readout.set_weights([readout.coef_.T, readout.intercept_])
        # self.model(transient_data[:, :1, :])

        self.model = keras.Model(
            inputs=self.reservoir.inputs,
            outputs=self.readout(self.reservoir.outputs[0]),
            name="ESN",
        )
    

    def train_test(
        self,
        transient_data: np.ndarray,
        train_data: np.ndarray,
        train_target: np.ndarray,
        regularization: int,
    ):
        if not self.reservoir:
            self.reservoir = keras.Model(
                inputs=self.inputs,
                outputs=self.outputs
            )
        
        print("\nEnsuring ESP...\n")
        if not self.reservoir.built:
            self.reservoir.build(input_shape=transient_data.shape)
        self.reservoir.predict(transient_data)

        print("\nHarvesting...\n")
        start = time.time()
        harvested_states = self.reservoir.predict(train_data)
        end = time.time()
        print(f"Harvesting took: {round(end - start, 2)} seconds.")
        
        print("Calculating the readout matrix...\n")

        readout_matrix, readout_bias = tf_ridge_regression(
            harvested_states[0], train_target[0], regularization, 'svd'
        )

        readout_layer = self.readout

        readout_layer.build(harvested_states[0].shape)

        # Applying the readout weights
        readout_layer.set_weights([readout_matrix, readout_bias])

        # readout = Ridge(alpha=regularization, tol=0, solver=solver)

        # readout.fit(harvested_states[0], train_target[0])

        # Training error of the readout
        predicted = readout_layer(harvested_states[0])
        
        predicted_1 = tf.matmul(harvested_states[0], readout_matrix) + readout_bias
        
        print("Comparing the layer and the real stuff", tf.reduce_mean(predicted - predicted_1))

        training_loss = np.mean(np.abs((predicted - train_target[0])))

        print(f"Training loss: {training_loss}\n")
        
        # Show NRMSE of the readout with respect to the training data
        
        NRMSE = np.sqrt(np.mean(np.square(predicted - train_target[0]))) / np.std(train_target[0])
        print(f"NRMSE: {NRMSE}\n")

        self.model = keras.Model(
            inputs=self.reservoir.inputs,
            outputs=readout_layer(self.reservoir.outputs[0]),
            name="ESN",
        )


    def forecast(
            self,
            forecast_length: int,
            forecast_transient_data: np.ndarray,
            val_data: np.ndarray,
            val_target: np.ndarray,
        ):
        forecast_length = min(forecast_length, val_data.shape[1])
        _val_target = val_target[:, :forecast_length, :]

        print(f"Forecasting free running sequence {forecast_length} steps ahead.\n")
        print("    Ensuring ESP...\n")
        print("    Forecast transient data shape: ", forecast_transient_data.shape)
        self.predict(forecast_transient_data)
        
        predictions = val_data[:, :1, :]
        print("\n    Predicting...\n")
        for _ in track(range(forecast_length)):
            pred = self.model(predictions[:, -1:, :])
            predictions = np.hstack((predictions, pred))
        
        predictions = predictions[:, 1:, :]
        print("    Predictions shape: ", predictions.shape)

        try:
            loss = np.mean((predictions[0] - _val_target[0]) ** 2)
        except ValueError:
            print("Error calculating the loss.")
            return np.inf
        print(f"Forecast loss: {loss}\n")

        self.model.reset_states()

        return predictions

    def save(self, filepath: str):
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str):
        model: keras.Model = keras.models.load_model(filepath, compile=False)
        inputs = model.layers[0].output
        outputs = model.layers[-2].output
        readout = model.layers[-1]
        reservoir = keras.Model(
            inputs=inputs,
            outputs=outputs,
        )
        model = keras.Model(
            inputs=reservoir.inputs,
            outputs=readout(reservoir.outputs[0]),
            name="ESN",
        )
        esn = ESN(
            inputs,
            outputs,
            readout
        )
        esn.reservoir = reservoir
        esn.model = model
        return esn


def generate_ESN(
        units: int,
        leak_rate: float = 1.,
        features: int = 1,
        activation: str = 'tanh',
        input_reservoir_init="InputMatrix",
        input_bias_init="random_uniform",
        reservoir_kernel_init="WattsStrogatzNX",
        exponent=2,
        seed: int = None
    ):
    
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f'\nSeed: {seed}\n')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    inputs = keras.Input(batch_shape=(1, None, features), name='Input')

    esn_cell = EsnCell(
        units=units,
        name="EsnCell",
        activation=activation,
        leak_rate=leak_rate,
        input_initializer=input_reservoir_init,
        input_bias_initializer=input_bias_init,
        reservoir_initializer=reservoir_kernel_init,
    )

    esn_rnn = keras.layers.RNN(
        esn_cell,
        trainable=False,
        stateful=True,
        return_sequences=True,
        name="esn_rnn",
    )(inputs)

    power_index = PowerIndex(exponent=exponent, index=2, name="power_index")(
        esn_rnn
    )

    outputs = keras.layers.Concatenate(name="Concat_ESN_input")(
        [inputs, power_index]
    )

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ESN(inputs, outputs, readout_layer)
    return model
