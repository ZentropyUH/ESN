import tensorflow as tf
import numpy as np

class WarmupStatesCallback(tf.keras.callbacks.Callback):
    """
    On each epoch-begin, zero out all stateful RNN layers then
    run transient_data through model.predict(...) to evolve states.
    """
    def __init__(self, transient_data: np.ndarray, batch_size: int = 32) -> None:
        super().__init__()
        self.transient_data = transient_data
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None) -> None:
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        self.model.predict(
            self.transient_data,
            batch_size=self.batch_size,
            verbose=0
        )
