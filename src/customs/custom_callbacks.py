"""Custom keras callbacks."""
import tensorflow as tf
import keras


###############################################
################## Callbacks ##################
###############################################


# A blanck example to derive my own
@tf.keras.utils.register_keras_serializable(package="custom")
class CustomCallback(keras.callbacks.Callback):
    """An example of a custom callback with all the methods."""

    def on_train_begin(self, logs=None):
        """Call at the start of training."""
        keys = list(logs.keys())
        print(f"Starting training; got log keys: {keys}")

    def on_train_end(self, logs=None):
        """Call at the end of training."""
        keys = list(logs.keys())
        print(f"Stop training; got log keys: {keys}")

    def on_epoch_begin(self, epoch, logs=None):
        """Call at the start of an epoch."""
        keys = list(logs.keys())
        print(f"Start epoch {epoch} of training; got log keys: {keys}")

    def on_epoch_end(self, epoch, logs=None):
        """Call at the end of an epoch."""
        keys = list(logs.keys())
        print(f"End epoch {epoch} of training; got log keys: {keys}")

    def on_test_begin(self, logs=None):
        """Call at the start of evaluation."""
        keys = list(logs.keys())
        print(f"Start testing; got log keys: {keys}")

    def on_test_end(self, logs=None):
        """Call at the end of evaluation."""
        keys = list(logs.keys())
        print(f"Stop testing; got log keys: {keys}")

    def on_predict_begin(self, logs=None):
        """Call at the start of prediction."""
        keys = list(logs.keys())
        print(f"Start predicting; got log keys: {keys}")

    def on_predict_end(self, logs=None):
        """Call at the end of prediction."""
        keys = list(logs.keys())
        print(f"Stop predicting; got log keys: {keys}")

    def on_train_batch_begin(self, batch, logs=None):
        """Call right before processing a batch during training."""
        keys = list(logs.keys())
        print(f"...Training: start of batch {batch}; got log keys: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        """Call at the end of training a batch."""
        keys = list(logs.keys())
        print(f"...Training: end of batch {batch}; got log keys: {keys}")

    def on_test_batch_begin(self, batch, logs=None):
        """Call right before processing a batch during evaluation."""
        keys = list(logs.keys())
        print(f"...Evaluating: start of batch {batch}; got log keys: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        """Call at the end of evaluation a batch."""
        keys = list(logs.keys())
        print(f"...Evaluating: end of batch {batch}; got log keys: {keys}")

    def on_predict_batch_begin(self, batch, logs=None):
        """Call right before processing a batch during prediction."""
        keys = list(logs.keys())
        print(f"...Predicting: start of batch {batch}; got log keys: {keys}")

    def on_predict_batch_end(self, batch, logs=None):
        """Call at the end of prediction a batch."""
        keys = list(logs.keys())
        print(f"...Predicting: end of batch {batch}; got log keys: {keys}")

    def get_config(self):
        """Return the config of the callback."""
        return {}


class ResetStatesCallback(keras.callbacks.Callback):
    """Callback to reset the states of the model."""

    def on_epoch_begin(self, epoch, logs=None):
        """Reset the states of the model."""
        self.model.reset_states()


class ThermalizeNetCallback(keras.callbacks.Callback):
    """Callback to thermalize the model calling predicting a transient data."""

    def __init__(self, thermalize_data, **kwargs):
        """Initialize the callback.

        Args:
            thermalize_data (np.ndarray): Data to thermalize the model.
                This is the transient to eliminate initial state of the RNN.
        """
        self.thermalize_data = thermalize_data
        super(ThermalizeNetCallback, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        """Reset the states of the model."""
        self.model.predict(self.thermalize_data)

    def on_epoch_end(self, epoch, logs=None):
        """Reset the states of the model."""
        self.model.reset_states()
