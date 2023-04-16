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
