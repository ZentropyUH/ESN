"""Custom keras layers."""
from typing import Dict, List, Union

import keras
import tensorflow as tf


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


@keras.saving.register_keras_serializable(package="MyLayers", name="RemoveOutliersAndMean")
class RemoveOutliersAndMean(keras.layers.Layer):
    """Removes the outliers from the input tensor and computes the mean of the remaining elements. We use the default threshold of 3.0 for both methods, based on Chebyshev's inequality, which states that at least 88.9% of the data lies within 3 standard deviations of the mean. So, roughly 11.1% of the data can be considered as outliers in worst case scenario.

    Args:
        method (str): The method to remove the outliers. Can be 'z_score' or 'iqr'. Default is 'z_score'.
        threshold (float): The threshold to remove the outliers. Default is 3.0.

    Returns:
        keras.layers.Layer: A keras layer that removes the outliers from the input tensor and computes the mean of the remaining elements.
    """
    def __init__(self, method='z_score', threshold=3.0, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold

    def build(self, input_shape) -> None:
        """Build the layer.

        Args:
            input_shape: The shape of the input tensor.
        """
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        super().build(input_shape)

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]]
    ) -> tf.Tensor:
        """Receives a 2D tensor and removes the outliers over the first dimension using the method provided and computes the mean of the remaining elements.

        Args:
            inputs (Union[tf.Tensor, List[tf.Tensor]]): Either a tensor of shape (samples, features) or a list of tensors of shape (features,). If a list is provided, the tensors are stacked along the first dimension to form a tensor of shape (samples, features).

        Returns:
            tf.Tensor: Mean of the remaining elements after removing the outliers. Of shape (1, features).
        """
        if isinstance(inputs, list):
            shape = inputs[0].shape
            for tensor in inputs:
                assert tensor.shape == shape, "All tensors must have the same shape"

            if len(shape) == 3:
                for i, tensor in enumerate(inputs):
                    inputs[i] = tf.squeeze(tensor, axis=0)

            inputs = tf.concat(inputs, axis=0)

        else:
            # The input tensor is of shape (1, samples, features), we need to strip the batch dimension
            if inputs.ndim == 3:
                inputs = tf.squeeze(inputs, axis=0)

        if self.method == 'z_score':
            mean = keras.ops.mean(inputs, axis=0, keepdims=True)
            std = keras.ops.std(inputs, axis=0, keepdims=True)

            # Avoid division by zero
            std_safe = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

            z_scores = keras.ops.abs((inputs - mean) / std_safe)
            mask = z_scores < self.threshold

        elif self.method == 'iqr':
            q25, q75 = keras.ops.quantile(inputs, 0.25), keras.ops.quantile(inputs, 0.75)
            iqr = q75 - q25
            lower = q25 - self.threshold * iqr
            upper = q75 + self.threshold * iqr
            mask = keras.ops.logical_and(inputs > lower, inputs < upper)

        output = keras.ops.multiply(inputs, mask)
        output = keras.ops.mean(output, axis=0, keepdims=True)

        # Add the batch dimension back
        inputs = tf.expand_dims(output, axis=0)
        return output

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape

    def get_config(self):
        config = super(RemoveOutliersAndMean, self).get_config()
        config.update({
            'method': self.method,
            'threshold': self.threshold
        })
        return config


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

