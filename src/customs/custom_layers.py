"""Custom keras layers."""
from typing import Dict, List

import tensorflow as tf
import keras
import keras.utils
import keras.layers
import keras.initializers
import keras.activations


@keras.saving.register_keras_serializable(package="MyLayers", name="PowerIndex")
class PowerIndex(keras.layers.Layer):
    """Applies a power function to the input even/odd indexed elements.

    Index can be an integer, depending on its parity will power the
    corresponding elements of the same parity from the input.

    Args:
        index (int): The index of the power function.

        exponent (float): The exponent of the power function.

    Returns:
        keras.layers.Layer: A keras layer that applies a power function to the
            input elements of the same parity as index.

    #### Example usage:

    >>> layer = PowerIndex(index=2, exponent=2)
    >>> layer(tf.constant([1, 2, 3, 4]))
    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 1,  4,  3, 16], dtype=int32)>
    """

    def __init__(self, index, exponent, **kwargs) -> None:
        """Initialize the layer."""
        super().__init__(**kwargs)
        self.index = (index) % 2
        self.exponent = exponent

    def call(self, inputs) -> tf.Tensor:
        """Compute the output tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor with the elements of the same
                parity of index powered by exponent.
        """
        dim = tf.shape(inputs)[-1]

        mask = tf.math.mod(tf.range(dim), 2)

        if self.index:
            mask = 1 - mask

        masked = tf.math.multiply(tf.cast(mask, tf.float32), inputs)

        unmaksed = tf.math.multiply(1 - tf.cast(mask, tf.float32), inputs)

        output = tf.math.pow(masked, self.exponent) + unmaksed

        return output

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = super().get_config()
        config.update({"index": self.index, "exponent": self.exponent})
        return config

    def get_weights(self) -> List:
        """Return the weights of the layer."""
        return []


# For the ParallelReservoir model
@keras.saving.register_keras_serializable(package="MyLayers", name="InputSplitter")
class InputSplitter(keras.layers.Layer):
    def __init__(self, partitions, overlap, **kwargs):
        super().__init__(**kwargs)
        self.partitions = partitions
        self.overlap = overlap

    def call(self, inputs):
        # Handling the case when partitions are 1
        if self.partitions == 1:
            return [inputs]

        # Shape validation
        batch_size, sequence_length, features = inputs.shape
        assert features % self.partitions == 0, "Feature dimension must be divisible by partitions"
        assert features // self.partitions + 1 > self.overlap, "Overlap must be smaller than the length of the partitions."



        # Calculating the width of each partition including overlap
        partition_width = features // self.partitions + 2 * self.overlap

        # Applying circular wrapping
        wrapped_inputs = tf.concat([inputs[:, :, -self.overlap:], inputs, inputs[:, :, :self.overlap]], axis=-1)

        # Slicing the input tensor into partitions
        partitions = []
        for i in range(self.partitions):
            start = i * (features // self.partitions)
            end = start + partition_width
            partitions.append(wrapped_inputs[:, :, start:end])

        return partitions

    def compute_output_shape(self, input_shape):
        batch_size, sequence_length, features = input_shape
        partition_width = features // self.partitions + 2 * self.overlap
        return [(batch_size, sequence_length, partition_width) for _ in range(self.partitions)]

    def get_config(self):
        config = super(InputSplitter, self).get_config()
        config.update({
            'partitions': self.partitions,
            'overlap': self.overlap
        })
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

