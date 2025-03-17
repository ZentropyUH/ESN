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

    The layer is designed to selectively exponentiate specific features in a structured way.

    **Behavior**:
        - If ``index`` is even, the layer exponentiates **even** positions in the last dimension.
        - If ``index`` is odd, the layer exponentiates **odd** positions in the last dimension.
        - The remaining elements are left unchanged.

    Parameters
    ----------
    index : int
        Integer index used solely for determining whether to exponentiate odd or even indices
        in the last dimension. The layer internally uses ``index % 2``.
    exponent : float
        The exponent to which the selected positions will be raised.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    index : int
        The stored integer index used to determine parity.
    exponent : float
        The stored exponent value used to transform selected elements.

    Input Shape
    -----------
    (batch, ..., features)

    Output Shape
    ------------
    (batch, ..., features) (same as input)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.custom_layers import SelectiveExponentiation
    >>> layer = SelectiveExponentiation(index=2, exponent=2.0)
    >>> x = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    >>> y = layer(x)
    >>> print(y.numpy())
    [[ 1.  4.  3. 16.]]
    """


    def __init__(self, index: int, exponent: float, **kwargs) -> None:
        super().__init__(**kwargs)
        # Determine parity only once
        self.index = index
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

        # Mask for even/odd indices. Will be 1 where indexes have the same parity as self.index, 0 otherwise.
        mask = tf.cast(tf.math.equal(tf.math.mod(tf.range(dim), 2), (self.index + 1) % 2), tf.float32)

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
    then returns the mean of the remaining elements.

    This layer is useful for denoising temporal data by filtering out extreme values.

    **Input Shape**:
        ``(samples, batch, timesteps, features)``

    **Output Shape**:
        ``(batch, timesteps, features)``

    **Procedure**:
        1. Compute the L2 norm over the last dimension (features), resulting in shape
           ``(samples, batch, timesteps)``.
        2. For each `(batch, timestep)`, compute outlier thresholds using:
           - **Z-score method**: mean and std deviation across `samples`.
           - **IQR method**: first and third quartiles across `samples`.
        3. Build a mask indicating inlier vs. outlier samples.
        4. Compute the mean of the inlier samples.
        5. If an entire `(batch, timestep)` has no valid samples, fallback to numerical stability.

    Parameters
    ----------
    method : str, optional
        Outlier removal method. Choices: ``{"z_score", "iqr"}``. Defaults to ``"z_score"``.
    threshold : float, optional
        Threshold for removing outliers (e.g., 3.0 for ±3 std if using Z-score). Defaults to 3.0.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    method : str
        The chosen outlier detection method.
    threshold : float
        The threshold value used for filtering.

    Raises
    ------
    ValueError
        If `method` is not one of ``"z_score"`` or ``"iqr"``.
        If using IQR method, TensorFlow Probability (`tfp`) must be installed.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.custom_layers import OutliersFilteredMean
    >>> layer = OutliersFilteredMean(method="z_score", threshold=2.0)
    >>> x = tf.random.normal((10, 3, 5, 4))  # 10 samples, batch=3, timesteps=5, features=4
    >>> y = layer([x,x,x])
    >>> print(y.shape)
    (3, 5, 4)
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

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
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
        if isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=0)  # (samples, batch, None, features)
        else:
            inputs = tf.expand_dims(inputs, axis=0)  # (1, batch, None, features)

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
            q1 = keras.ops.quantile(norms, 0.25, axis=0)
            q3 = keras.ops.quantile(norms, 0.75, axis=0)
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
        count_ = tf.broadcast_to(count_, tf.shape(sum_))  # (batch, timesteps, features)

        # Avoid dividing by zero: if count_ is 0, replace with 1
        count_ = tf.where(count_ > 0, count_, tf.ones_like(count_))

        # 6) Take the mean => shape (batch, timesteps, features)
        mean_ = sum_ / count_

        return mean_

    def compute_output_shape(self, input_shape):
        # If input is a list, use one element’s shape (they all have the same shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]  # Single tensor shape: (batch, timesteps, features)

        # Ensure it's the expected format (batch, timesteps, features) before returning
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]])

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method, "threshold": self.threshold})
        return config


# For the ParallelReservoir model
@keras.saving.register_keras_serializable(package="krc", name="FeaturePartitioner")
class FeaturePartitioner(keras.layers.Layer):
    r"""
    A Keras layer that partitions the feature dimension into multiple overlapping slices,
    with optional circular wrapping at the boundaries.

    This layer is useful for dividing input features into structured regions while maintaining
    smooth transitions between partitions.

    **Behavior**:
        - Splits the feature dimension into `partitions` groups.
        - Each partition overlaps with its neighbors by `overlap` units.
        - Optionally applies **circular wrapping**, where the last `overlap` features wrap around
          to the start, and vice versa.

    Parameters
    ----------
    partitions : int
        Number of partitions to divide the feature dimension into.
    overlap : int
        The overlap size (in feature units) for each partition.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    partitions : int
        The number of partitions.
    overlap : int
        The overlap size for each partition.

    Input Shape
    -----------
    (batch_size, sequence_length, features)

    Output Shape
    ------------
    List of ``partitions`` tensors, each of shape
    ``(batch_size, sequence_length, partition_width)``, where
    ``partition_width = features // partitions + 2 * overlap``.

    Raises
    ------
    ValueError
        If `features % partitions != 0` (unless `partitions == 1`).
        If `features // partitions + 1 <= overlap` (invalid overlap size).

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.custom_layers import FeaturePartitioner
    >>> layer = FeaturePartitioner(partitions=2, overlap=1)
    >>> x = tf.reshape(tf.range(12), (1, 1, 12))  # shape (batch=1, seq_len=1, features=12)
    >>> outputs = layer(x)
    >>> len(outputs)
    2
    >>> print(outputs[0].numpy())
    [[11 0 1 2 3 4 5 6]]
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


@keras.saving.register_keras_serializable(package="krc", name="SelectiveDropout")
class SelectiveDropout(keras.layers.Layer):
    """
    A Keras layer that zeroes out specific features across all timesteps and batches,
    based on a fixed mask provided at initialization.

    This layer is useful for analyzing how shutting off specific neurons affects model predictions.

    Parameters
    ----------
    mask : array-like or tf.Tensor, shape (features,)
        A 1D boolean mask where `True` indicates that the corresponding feature should always be zeroed out.

    Attributes
    ----------
    mask : tf.Tensor
        The stored boolean mask tensor of shape (features,).

    Input Shape
    -----------
    (batch, timesteps, features)

    Output Shape
    ------------
    (batch, timesteps, features)

    Examples
    --------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.custom_layers import SelectiveDropout
    >>> mask = np.array([False, True, False, True])  # Drop feature indices 1 and 3
    >>> x = np.random.rand(2, 5, 4)
    >>> layer = SelectiveDropout(mask)
    >>> # Apply the layer
    >>> y = layer(x)
    >>> print(y.numpy())  # Features 1 and 3 should be zeroed out across all timesteps and batches
    """

    def __init__(self, mask: tf.Tensor, **kwargs):
        """
        Initializes the SelectiveDropout layer with a fixed mask.

        Parameters
        ----------
        mask : tf.Tensor, shape (features,)
            A 1D boolean mask indicating which features should be zeroed out.
        """
        super().__init__(**kwargs)

        # Convert mask to tensor and ensure it's a boolean tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.bool)

        # Ensure mask is strictly 1D
        if mask.ndim != 1:
            raise ValueError(f"Mask must be a 1D tensor, but got shape {mask.shape}")

        self.mask = mask

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Ensures the mask size matches the feature dimension.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor, expected to be (batch, timesteps, features).

        Raises
        ------
        ValueError
            If the mask size does not match the feature dimension.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch, timesteps, features), but got {input_shape}"
            )

        feature_dim = input_shape[-1]
        if self.mask.shape[0] != feature_dim:
            raise ValueError(
                f"Mask size {self.mask.shape[0]} does not match feature dimension {feature_dim}"
            )

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Applies selective dropout using the stored mask.

        Parameters
        ----------
        inputs : tf.Tensor
            Input data of shape (batch, timesteps, features).
        training : bool, optional
            Training flag (not explicitly used but kept for compatibility).

        Returns
        -------
        tf.Tensor
            The input tensor with masked features set to zero.
        """
        return tf.where(self.mask, 0.0, inputs)

    def compute_output_shape(
        self, input_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """
        Computes the output shape, which is identical to the input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            The same shape as the input.
        """
        return input_shape

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns
        -------
        dict
            A dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"mask": self.mask})
        return config
