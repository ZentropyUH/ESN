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
    """Removes the outliers from the input tensor and computes the mean of the remaining elements.

    We use the default threshold of 3.0 for both methods, based on Chebyshev's inequality, which states that at least 88.9% of the data lies within 3 standard deviations of the mean. So, roughly 11.1% of the data can be considered as outliers in worst case scenario.

    Args:
        method (str): The method to remove the outliers. Can be 'z_score' or 'iqr'. Default is 'z_score'.
        threshold (float): The threshold to remove the outliers. Default is 3.0.

    Returns:
        keras.layers.Layer: A keras layer that removes the outliers from the input tensor and computes the mean of the remaining elements.
    """

    def __init__(self, method="z_score", threshold=3.0, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold

    def build(self, input_shape) -> None:
        """Build the layer.

        Args:
            input_shape: The shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs):
        """Remove the outliers from the input tensor and compute the mean of the remaining elements.

        The method removes the outliers from the input tensor according to 'method' and computes the mean of the remaining elements.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor of shape (samples, batch, sequence_length, features).
            The outliers will be taken along the first dimension (samples) according to the method respect to the norm of the vectors along the last dimension (features).
        Returns
        -------
        tf.Tensor
            The output tensor of shape (batch, sequence_length, features) with the mean of the remaining elements.
        """

        # Calculate the norm of each vector along the last dimension
        norms = tf.norm(inputs, axis=-1)

        if self.method == "z_score":
            # Calculate the mean and standard deviation of the norms
            mean_norm = tf.reduce_mean(norms, axis=0)
            std_norm = tf.math.reduce_std(norms, axis=0)

            # Safety check to avoid division by zero
            std_norm = tf.where(
                std_norm > 0, std_norm, tf.ones_like(std_norm)
            )  # Match shape

            # Identify non-outlier indices using the Z-score method
            z_scores = tf.abs((norms - mean_norm) / std_norm)

            non_outlier_mask = z_scores < self.threshold

        elif self.method == "iqr":
            # Calculate the first quartile (Q1) and third quartile (Q3)
            q1 = keras.ops.quantile(norms, 0.25, axis=0)
            q3 = keras.ops.quantile(norms, 0.75, axis=0)

            # Calculate the Interquartile Range (IQR)
            iqr = q3 - q1

            # Identify non-outlier indices using the IQR method
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            non_outlier_mask = (norms >= lower_bound) & (norms <= upper_bound)

        else:
            raise ValueError(
                f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'."
            )

        mask_1d = tf.reduce_any(
            non_outlier_mask, axis=[1, 2]
        )  # Reduce over batch & sequence_length

        # Filter out the outliers
        filtered_inputs = tf.boolean_mask(inputs, mask_1d, axis=0)

        # Compute the mean of the remaining vectors
        result = tf.reduce_mean(
            filtered_inputs, axis=0
        )  # should be of shape (batch, sequence_length, features)

        return result

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape of shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        return input_shape[1:]

    def get_config(self):
        config = super(RemoveOutliersAndMean, self).get_config()
        config.update({"method": self.method, "threshold": self.threshold})
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
