"""
Custom Keras layers.

This module contains custom Keras layers that are not part of the standard Keras library. 
The layers are used as part of the Reservoir Computing models in this package.
"""

from typing import Dict, List, Union

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(
    package="krc", name="SelectiveExponentiation"
)
class SelectiveExponentiation(keras.layers.Layer):
    r"""
    A Keras layer that exponentiates either the even or odd indices of the last dimension,
    depending on the parity of a given integer index.

    **Behavior**:
        - If ``index`` is even, the layer exponentiates **even** positions in the last dimension.
        - If ``index`` is odd, the layer exponentiates **odd** positions in the last dimension.
        - The rest of the elements are left unchanged.

    Parameters
    ----------
    index : int
        Integer index used solely for determining whether to exponentiate odd or even indices
        in the last dimension. The code internally uses ``index % 2``.
    exponent : float
        The exponent to which the selected positions will be raised.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from my_layers import PowerIndex

    >>> layer = PowerIndex(index=2, exponent=2.0)
    >>> x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    >>> layer(x)
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 1.,  4.,  3., 16.], dtype=float32)>
    """

    def __init__(self, index: int, exponent: float, **kwargs) -> None:
        super().__init__(**kwargs)
        # Determine parity only once
        self.index = (index + 1) % 2
        self.exponent = exponent

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        r"""
        Exponentiates either even or odd indices (in the last dimension) based on ``index`` parity.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor. Can be of any shape, but the exponentiation mask is applied
            along the last dimension.

        Returns
        -------
        tf.Tensor
            The output tensor, where either even or odd positions have been exponentiated
            by ``self.exponent``, depending on ``index`` parity.
        """
        dim = tf.shape(inputs)[-1]
        
        # mask = 0 for even indices, 1 for odd indices
        mask = tf.math.mod(tf.range(dim), 2)

        # If index is odd => invert the mask => exponentiate odd indices
        mask = self.index - mask

        mask_f = tf.cast(mask, dtype=tf.float32)

        # Elements to be exponentiated
        masked = inputs * mask_f
        # Elements to remain the same
        unmasked = inputs * (1.0 - mask_f)

        output = tf.math.pow(masked, self.exponent) + unmasked
        return output

    def compute_output_shape(
        self, input_shape: Union[tf.TensorShape, List[int]]
    ) -> tf.TensorShape:
        r"""
        Computes the output shape, which is the same as the input shape.

        Parameters
        ----------
        input_shape : tf.TensorShape or list of int
            The shape of the input tensor.

        Returns
        -------
        tf.TensorShape
            The same shape as ``input_shape``.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict:
        r"""
        Returns the configuration of the layer for serialization.

        Returns
        -------
        dict
            A dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"index": self.index, "exponent": self.exponent})
        return config


@keras.saving.register_keras_serializable(
    package="krc", name="OutliersFilteredMean"
)
class OutliersFilteredMean(keras.layers.Layer):
    r"""
    A Keras layer that removes outliers (along the `samples` dimension) independently at
    each (batch, timestep) location, based on a specified method (Z-score or IQR), and
    then returns the mean of the remaining elements (in the `samples` dimension).

    **Input shape**:
        ``(samples, batch, timesteps, features)``

    **Output shape**:
        ``(batch, timesteps, features)``

    **Procedure**:
        1. Compute the L2 norm over the last dimension (features), resulting in shape
           ``(samples, batch, timesteps)``.
        2. For each `(batch, timestep)` pair, compute either:
           - **Z-score method**: mean and std of the norms across `samples`.
           - **IQR method**: Q1 and Q3 of the norms across `samples`.
        3. Build a boolean mask indicating which samples are inliers vs. outliers
           (per `(batch, timestep)`).
        4. Use this mask to include/exclude specific samples when computing the final mean.
        5. The final mean is taken across the `samples` dimension, resulting in
           ``(batch, timesteps, features)``. If an entire `(batch, timestep)` ends up with
           no valid samples, the code avoids division by zero by forcing a 1 in the denominator.

    Parameters
    ----------
    method : str, optional
        Outlier removal method. ``{"z_score", "iqr"}``. Defaults to ``"z_score"``.
    threshold : float, optional
        Threshold for removing outliers (e.g., 3.0 for ±3 std if using Z-score). Defaults to 3.0.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Raises
    ------
    ValueError
        If `method` is not one of ``"z_score"`` or ``"iqr"``
        If using IQR method, TensorFlow Probability (`tfp`) must be installed.
    """

    def __init__(
        self, method: str = "z_score", threshold: float = 3.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold

        if self.method not in ["z_score", "iqr"]:
            raise ValueError(
                f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'."
            )

    def build(self, input_shape):
        # No trainable parameters to build
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Removes outliers at each (batch, timestep), then averages over the samples dimension.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape ``(samples, batch, timesteps, features)``.

        Returns
        -------
        tf.Tensor
            A tensor of shape ``(batch, timesteps, features)``, representing the mean of the
            non-outlier samples.
        """
        # 1) Compute the norm over the last dimension => shape (samples, batch, timesteps)
        norms = tf.norm(inputs, axis=-1)

        # 2) For each (batch, timestep), figure out which samples are inliers vs outliers
        if self.method == "z_score":
            # mean_norm, std_norm => shape (batch, timesteps)
            mean_norm = tf.reduce_mean(norms, axis=0)
            std_norm = tf.math.reduce_std(norms, axis=0)

            # Avoid division by zero by forcing std_norm to 1 where it's 0
            std_norm = tf.where(std_norm > 0, std_norm, tf.ones_like(std_norm))

            # z-scores => shape (samples, batch, timesteps)
            z_scores = tf.abs((norms - mean_norm) / std_norm)

            # True => sample is inlier, False => outlier
            mask = z_scores < self.threshold

        else:  # self.method == "iqr"
            # Q1, Q3 => shape (batch, timesteps)
            q1 = keras.ops.quantile(norms, 0.25, axis=0, interpolation="linear")
            q3 = keras.ops.quantile(norms, 0.75, axis=0, interpolation="linear")
            iqr = q3 - q1

            # Lower/upper bounds
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr

            # True => inlier, False => outlier
            mask = (norms >= lower_bound) & (norms <= upper_bound)

        # mask shape => (samples, batch, timesteps)
        # We need this mask to broadcast along the features dimension for the final averaging.
        mask_expanded = tf.cast(tf.expand_dims(mask, axis=-1), dtype=inputs.dtype)
        # mask_expanded => (samples, batch, timesteps, 1)

        # 3) Multiply inputs by mask to zero out outliers
        masked_inputs = inputs * mask_expanded
        # shape => (samples, batch, timesteps, features)

        # 4) Sum along the samples dimension => shape (batch, timesteps, features)
        sum_ = tf.reduce_sum(masked_inputs, axis=0)

        # 5) Count inlier samples at each (batch, timestep) => shape (batch, timesteps, 1)
        count_ = tf.reduce_sum(mask_expanded, axis=0, keepdims=False)
        count_ = tf.expand_dims(count_, axis=-1)  # shape => (batch, timesteps, 1)

        # Avoid dividing by zero: if count_ is 0, replace with 1
        count_ = tf.where(count_ > 0, count_, tf.ones_like(count_))

        # 6) Take the mean => shape (batch, timesteps, features)
        mean_ = sum_ / count_

        return mean_

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """
        Output shape is (batch, timesteps, features).
        """
        # input_shape => (samples, batch, timesteps, features)
        return tf.TensorShape([input_shape[1], input_shape[2], input_shape[3]])

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method, "threshold": self.threshold})
        return config


# For the ParallelReservoir model
@keras.saving.register_keras_serializable(package="krc", name="FeaturePartitioner")
class FeaturePartitioner(keras.layers.Layer):
    r"""
    A Keras layer that splits the feature dimension into multiple overlapping partitions,
    with optional circular wrapping on both ends.

    Given an input of shape ``(batch_size, sequence_length, features)``, this layer:

    - Validates that ``features`` is divisible by ``partitions`` (unless ``partitions == 1``).
    - Optionally overlaps the partitions by ``overlap`` units on each side.
    - Performs circular wrapping by concatenating the last ``overlap`` features to the front
      and the first ``overlap`` features to the end, then slicing out each partition.

    Parameters
    ----------
    partitions : int
        Number of partitions to split the feature dimension into.
    overlap : int
        The overlap (in feature dimension units) on each side for each partition.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from my_layers import InputSplitter

    >>> layer = InputSplitter(partitions=2, overlap=1)
    >>> x = tf.reshape(tf.range(12), (1, 1, 12))  # shape (batch=1, seq_len=1, features=12)
    >>> outputs = layer(x)
    >>> len(outputs)
    2
    >>> outputs[0].shape
    TensorShape([1, 1, 8])

    Raises
    ------
    AssertionError
        If ``features // partitions + 1 <= overlap`` (invalid overlap).
    AssertionError
        If ``features % partitions != 0`` (unless ``partitions == 1``).
    """

    def __init__(self, partitions: int, overlap: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.partitions = partitions
        self.overlap = overlap

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        r"""
        Splits the feature dimension into overlapping partitions with circular wrapping.

        If ``self.partitions == 1``, returns a single list element containing ``inputs``.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape ``(batch_size, sequence_length, features)``.

        Returns
        -------
        list of tf.Tensor
            A list of length ``self.partitions``, each of shape
            ``(batch_size, sequence_length, partition_width)``,
            where ``partition_width = features // partitions + 2 * overlap``.
        """
        # If partitions == 1, just return the entire input as a single partition
        if self.partitions == 1:
            return [inputs]

        batch_size, sequence_length, features = inputs.shape
        # Validate shape
        assert (
            features % self.partitions == 0
        ), f"Feature dimension must be divisible by partitions when partitions > 1. Features are {features} and partitions are {self.partitions}."
        assert (
            features // self.partitions
        ) > self.overlap, "Overlap must be smaller than the length of each partition."

        # Width of each partition including overlap
        partition_width = (features // self.partitions) + 2 * self.overlap

        # Circular wrapping
        if self.overlap > 0:
            wrapped_inputs = tf.concat(
                [inputs[..., -self.overlap :], inputs, inputs[..., : self.overlap]],
                axis=-1,
            )
        else:
            wrapped_inputs = inputs

        partitions_out = []
        for i in range(self.partitions):
            start = i * (features // self.partitions)
            end = start + partition_width
            partitions_out.append(wrapped_inputs[..., start:end])

        return partitions_out

    def compute_output_shape(
        self, input_shape: Union[tf.TensorShape, List[int]]
    ) -> List[tf.TensorShape]:
        r"""
        Computes the output shapes for each partition.

        Parameters
        ----------
        input_shape : tf.TensorShape or list of int
            The shape of the input, typically ``(batch_size, sequence_length, features)``.

        Returns
        -------
        list of tf.TensorShape
            A list of shapes for each partition. Each shape is
            ``(batch_size, sequence_length, partition_width)``.
        """
        batch_size, sequence_length, features = input_shape
        partition_width = (features // self.partitions) + 2 * self.overlap
        return [
            tf.TensorShape([batch_size, sequence_length, partition_width])
            for _ in range(self.partitions)
        ]

    def get_config(self) -> Dict:
        r"""
        Returns the configuration of the layer for serialization.

        Returns
        -------
        dict
            A dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"partitions": self.partitions, "overlap": self.overlap})
        return config
