import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from time import time
from rich.progress import track
from typing import Any
from typing import Tuple
from typing import Optional

import keras
import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from src.customs.custom_layers import simple_esn
from src.customs.custom_layers import parallel_esn
from src.utils import calculate_nrmse


# TODO: Add log
# TODO: separate prediction and evaluation
class ESN:
    '''
    Base ESN model.\n
    Works as an assembly class for the layers that are passed to it as input.
    Groups a set of basic functionalities for an ESN.

    Args:
        inputs (keras.layers.Layer): The input layer for the ESN model.

        outputs (keras.layers.Layer): The model output layer. Set the logic of the ESN, until this point the model is not trained.

        readout (keras.layers.Layer): The layer that will be trained and will become the model output.
    '''
    def __init__(
        self,
        reservoir: Layer,
        readout: Layer,
    ) -> None:

        
        self.readout: Layer = readout
        self.reservoir: keras.layers.Layer = reservoir
        self.model: keras.Model = None
        
        self.built = False
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model is None:
            raise Exception('Model must be trained to predict')
        return self.model(*args, **kwds)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Predicts the output of the model for the given inputs.

        Args:
            inputs (np.ndarray): The input data to predict the output for.

        Returns:
            np.ndarray: The predicted output for the given inputs.

        Raises:
            Exception: If the model has not been trained yet.
        '''
        if self.model is None:
            raise Exception('Model must be trained to predict')
        return self.model.predict(inputs)
    
    def train(
        self,
        transient_data: np.ndarray,
        train_data: np.ndarray,
        train_target: np.ndarray,
        regularization: float,
    ) -> None:
        '''
        Training proccess of the model.

        Args:
            transient_data (np.ndarray): Transient data.

            train_data (np.ndarray): Data to train the model.

            train_target (np.ndarray): Target data for the training.

            regularization (float): Regularization value for linear readout.
        
        Return:
            None
        '''
        
        print("\nEnsuring ESP...\n")
        if not self.reservoir.built:
            self.reservoir.build(input_shape=transient_data.shape)
        self.reservoir.predict(transient_data)

        print("\nHarvesting...\n")
        start = time()
        harvested_states = self.reservoir.predict(train_data)
        end = time()
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

        nrmse = calculate_nrmse(target=train_target[0], prediction=predictions)
        print(f"NRMSE: {nrmse}\n")

        self.readout.build(harvested_states[0].shape)
        self.readout.set_weights([readout.coef_.T, readout.intercept_])
        # self.model(transient_data[:, :1, :])

        self.model = keras.Model(
            inputs=self.reservoir.inputs,
            # CHECK
            outputs=self.readout(self.reservoir.output),
            name="ESN",
        )
        

    # def train_test(
    #     self,
    #     transient_data: np.ndarray,
    #     train_data: np.ndarray,
    #     train_target: np.ndarray,
    #     regularization: int,
    # ):
    #     if not self.reservoir:
    #         self.reservoir = keras.Model(
    #             inputs=self.inputs,
    #             outputs=self.outputs
    #         )
        
    #     print("\nEnsuring ESP...\n")
    #     if not self.reservoir.built:
    #         self.reservoir.build(input_shape=transient_data.shape)
    #     self.reservoir.predict(transient_data)

    #     print("\nHarvesting...\n")
    #     start = time()
    #     harvested_states = self.reservoir.predict(train_data)
    #     end = time()
    #     print(f"Harvesting took: {round(end - start, 2)} seconds.")
        
    #     print("Calculating the readout matrix...\n")

    #     readout_matrix, readout_bias = tf_ridge_regression(
    #         harvested_states[0], train_target[0], regularization, 'svd'
    #     )

    #     readout_layer = self.readout

    #     readout_layer.build(harvested_states[0].shape)

    #     # Applying the readout weights
    #     readout_layer.set_weights([readout_matrix, readout_bias])

    #     # readout = Ridge(alpha=regularization, tol=0, solver=solver)

    #     # readout.fit(harvested_states[0], train_target[0])

    #     # Training error of the readout
    #     predicted = readout_layer(harvested_states[0])
        
    #     predicted_1 = tf.matmul(harvested_states[0], readout_matrix) + readout_bias
        
    #     print("Comparing the layer and the real stuff", tf.reduce_mean(predicted - predicted_1))

    #     training_loss = np.mean(np.abs((predicted - train_target[0])))

    #     print(f"Training loss: {training_loss}\n")
        
    #     # Show NRMSE of the readout with respect to the training data
        
    #     NRMSE = np.sqrt(np.mean(np.square(predicted - train_target[0]))) / np.std(train_target[0])
    #     print(f"NRMSE: {NRMSE}\n")

    #     self.model = keras.Model(
    #         inputs=self.reservoir.inputs,
    #         outputs=readout_layer(self.reservoir.outputs[0]),
    #         name="ESN",
    #     )

    
            # function code here

    @tf.function
    def forecast_step(self, model, current_input):
        """
        Forecast a single step using the model.
        Wrapped with tf.function for performance optimization.
        """
        return model(current_input, training=False)

    def forecast(
            self,
            forecast_length: int,
            forecast_transient_data: np.ndarray,
            val_data: np.ndarray,
            val_target: np.ndarray,
            internal_states: bool = False,
            feedback_metrics: bool = True
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        '''
        Forecast the model for a given number of steps.

        Args:
            forecast_length (int): Number of steps to forecast.

            forecast_transient_data (np.ndarray): Transient data of the val_data. The model is fitted with this data.

            val_data (np.ndarray): True data of the prediction. The forecast starts receiving its first value as input for the prediction.

            val_target (np.ndarray): Target data of the prediction. Used to calculate the loss function.

            internal_states (bool): Whether to return the states of the ESN.

            feedback_metrics: (bool): Whether to include comparison metrics with the original data.


        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: A tuple containing the forecasted data and the states of the ESN (if internal_states is True).
        '''
        self.model.reset_states()

        forecast_length = min(forecast_length, val_data.shape[1]) if feedback_metrics else forecast_length

        _val_target = val_target[:, :forecast_length, :]

        print(f"Forecasting free running sequence {forecast_length} steps ahead.\n")
        print("    Ensuring ESP...\n")
        print("    Forecast transient data shape: ", forecast_transient_data.shape)
        self.predict(forecast_transient_data)
        
        predictions = np.empty((val_data.shape[0], forecast_length + 1, val_data.shape[2]))
        predictions[:, 0:1, :] = val_data[:, :1, :]
        
        # Making the states an array of shape (0, units)
        states_over_time = np.empty((forecast_length, self.model.get_layer("esn_rnn").cell.units)) if internal_states else None

        print("\n    Predicting...\n")
        for step in track(range(forecast_length)):
            current_input = tf.convert_to_tensor(predictions[:, step:step + 1, :], dtype=tf.float32)
            pred = self.forecast_step(self.model, current_input)
            predictions[:, step + 1:step + 2, :] = pred
            
            if internal_states:
                # Getting the states of the ESN, also reducing the dimensionality
                #TODO: Make this generic to other type of reservoirs, not only simple_esn
                current_states = self.model.get_layer("esn_rnn").states[0].numpy()
                states_over_time[step, :] = current_states
        
        predictions = predictions[:, 1:, :]
        print("    Predictions shape: ", predictions.shape)

        if feedback_metrics:
            try:
                loss = np.mean((predictions[0] - _val_target[0]) ** 2)
                nrmse = calculate_nrmse(target=_val_target[0], prediction=predictions[0])
            except ValueError:
                print("Error calculating the loss.")
                return np.inf
            print(f"Forecast loss: {loss}\n")
            print(f"NRMSE: {nrmse}\n")

        return predictions, states_over_time

    def save(self, path: str) -> None:
        '''
        Save the model in a folder.

        Args:
            path (str): The destination folder to save all the files of the model.
        '''
        self.model.save(path, include_optimizer=False)
    
    # BUG: Some trained models cant be loaded. E.g. with bias InputMatrix works ok.
    @staticmethod
    def load(path: str) -> "ESN":
        '''
        Load the model from folder format.

        Args:
            path (str): Folder to load the model.
        
        Return:
            model (ESN): Return the loaded instance of the ESN model.
        '''
        model: keras.Model = keras.models.load_model(path, compile=False)
        inputs = model.get_layer("Input").output
        outputs = model.get_layer("Concat_ESN_input").output
        readout = model.get_layer("readout")
        reservoir = keras.Model(
            inputs=inputs,
            outputs=outputs,
        )
        model = keras.Model(
            inputs=reservoir.inputs,
            outputs=readout(reservoir.output),
            name="ESN",
        )
        esn = ESN(
            reservoir,
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
        input_reservoir_init: str = "InputMatrix",
        input_bias_init: str = "random_uniform",
        reservoir_kernel_init: str = "WattsStrogatzNX",
        exponent: int = 2,
        seed: int = None
    ) -> ESN:
    '''
    Assemble all the layers in ESN model.
    '''
    #TODO: see who handles the seed and how
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f'\nSeed: {seed}\n')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    reservoir = simple_esn(units=units,
                           leak_rate=leak_rate,
                           activation=activation,
                           features=features,
                           input_reservoir_init=input_reservoir_init,
                           input_bias_init=input_bias_init,
                           reservoir_kernel_init=reservoir_kernel_init,
                           exponent=exponent)

    readout_layer = Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ESN(reservoir, readout_layer)
    return model

def generate_Parallel_ESN(units: int,
                          partitions: int = 1,
                          overlap: int = 0,
                          leak_rate: float = 1.,
                          features: int = 1,
                          activation: str = 'tanh',
                          input_reservoir_init: str = "InputMatrix",
                          input_bias_init: str = "random_uniform",
                          reservoir_kernel_init: str = "WattsStrogatzNX",
                          exponent: int = 2,
                          seed: int = None
                        ) -> ESN:
    '''
    Assemble all the layers in a parallel ESN model.
    '''
    
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f'\nSeed: {seed}\n')
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    reservoir = parallel_esn(units=units,
                             partitions=partitions,
                             overlap=overlap,
                             leak_rate=leak_rate,
                             activation=activation,
                             features=features,
                             input_reservoir_init=input_reservoir_init,
                             input_bias_init=input_bias_init,
                             reservoir_kernel_init=reservoir_kernel_init,
                             exponent=exponent)
    
    readout_layer = Dense(
        features, activation="linear", name="readout", trainable=False
    )
    
    model = ESN(reservoir, readout_layer)
    return model
