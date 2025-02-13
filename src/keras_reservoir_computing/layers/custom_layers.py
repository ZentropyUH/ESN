"""Custom keras layers."""
from typing import Dict, List, Union

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="MyLayers", name="PowerIndex")
class PowerIndex(keras.layers.Layer):
    r"""
    A Keras layer that exponentiates either the even or odd indices of the last dimension,
    depending on the parity of a given integer index.

    **Behavior**:
        - If ``index`` is even, the layer exponentiates **odd** positions in the last dimension.
        - If ``index`` is odd, the layer exponentiates **even** positions in the last dimension.
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
        self.index = index % 2
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

        # If index is odd => invert the mask => exponentiate even indices
        if self.index == 1:
            mask = 1 - mask

        mask_f = tf.cast(mask, tf.float32)

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

    def get_weights(self) -> List:
        r"""
        This layer has no trainable parameters, so it returns an empty list.

        Returns
        -------
        list
            Empty list.
        """
        return []


@keras.saving.register_keras_serializable(
    package="MyLayers", name="RemoveOutliersAndMean"
)
class RemoveOutliersAndMean(keras.layers.Layer):
    r"""
    A Keras layer that removes outliers (along the first dimension) based on
    a specified method, then returns the mean of the remaining elements.

    The input is assumed to have shape ``(samples, batch, sequence_length, features)``.
    Outliers are determined by computing the norm over the last dimension (features),
    then applying one of the following:

    - **Z-score method** (``method="z_score"``):
      Removes any sample whose norm is further than ``threshold`` standard deviations
      from the mean norm (per ``batch, sequence_length``).
    - **IQR method** (``method="iqr"``):
      Removes any sample whose norm is outside the interval
      ``[Q1 - threshold * IQR, Q3 + threshold * IQR]``.

    After filtering, this layer computes the mean across the first dimension (the ``samples`` dimension),
    yielding an output of shape ``(batch, sequence_length, features)``.

    Parameters
    ----------
    method : str, optional
        The outlier removal method. Must be one of ``{"z_score", "iqr"}``. Default is ``"z_score"``.
    threshold : float, optional
        The threshold for removing outliers. Default is 3.0.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Raises
    ------
    ValueError
        If an unsupported ``method`` is provided.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from my_layers import RemoveOutliersAndMean

    >>> layer = RemoveOutliersAndMean(method="z_score", threshold=3.0)
    >>> x = tf.random.normal(shape=(10, 2, 5, 3))  # (samples, batch, seq_length, features)
    >>> out = layer(x)
    >>> out.shape
    TensorShape([2, 5, 3])
    """

    def __init__(
        self, method: str = "z_score", threshold: float = 3.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold

    def build(self, input_shape: Union[tf.TensorShape, List[int]]) -> None:
        r"""
        Builds the layer (no additional weights).

        Parameters
        ----------
        input_shape : tf.TensorShape or list of int
            The shape of the input tensor.
        """
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        r"""
        Removes outliers along the first dimension (``samples``) and computes the mean of
        the remaining elements.

        The norm of each sample (over the last dimension) is computed, and a mask is built
        according to the chosen method:

        - **Z-score**: based on the mean and standard deviation of norms.
        - **IQR**: based on quartiles (Q1, Q3) and the interquartile range (IQR).

        Any sample flagged as an outlier in **any** ``(batch, sequence_length)`` location is removed entirely.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape ``(samples, batch, sequence_length, features)``.

        Returns
        -------
        tf.Tensor
            A tensor of shape ``(batch, sequence_length, features)`` representing
            the mean of the non-outlier samples.
        """
        # Calculate the norm of each sample along the last dimension
        norms = tf.norm(inputs, axis=-1)  # (samples, batch, sequence_length)

        if self.method == "z_score":
            # Compute mean and std along the samples dimension=0, per (batch, sequence_length)
            mean_norm = tf.reduce_mean(norms, axis=0)
            std_norm = tf.math.reduce_std(norms, axis=0)

            # Safety check to avoid division by zero
            std_norm = tf.where(std_norm > 0, std_norm, tf.ones_like(std_norm))

            z_scores = tf.abs((norms - mean_norm) / std_norm)
            non_outlier_mask = z_scores < self.threshold

        elif self.method == "iqr":
            # Calculate Q1 and Q3 using Keras ops for quantiles
            q1 = keras.ops.quantile(norms, 0.25, axis=0)
            q3 = keras.ops.quantile(norms, 0.75, axis=0)
            iqr = q3 - q1

            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            non_outlier_mask = (norms >= lower_bound) & (norms <= upper_bound)

        else:
            raise ValueError(
                f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'."
            )

        # If a sample is an outlier at any (batch, sequence_length), it gets removed entirely.
        mask_1d = tf.reduce_any(non_outlier_mask, axis=[1, 2])  # shape (samples,)

        # Filter out the outlier samples
        filtered_inputs = tf.boolean_mask(inputs, mask_1d, axis=0)

        # Compute the mean of remaining samples
        result = tf.reduce_mean(
            filtered_inputs, axis=0
        )  # (batch, sequence_length, features)
        return result

    def compute_output_shape(
        self, input_shape: Union[tf.TensorShape, List[int]]
    ) -> tf.TensorShape:
        r"""
        Computes the output shape of the layer, which is ``(batch, sequence_length, features)``.

        Parameters
        ----------
        input_shape : tf.TensorShape or list of int
            The shape of the input tensor, expected as ``(samples, batch, sequence_length, features)``.

        Returns
        -------
        tf.TensorShape
            The shape ``(batch, sequence_length, features)``.
        """
        # input_shape is (samples, batch, sequence_length, features)
        return tf.TensorShape(input_shape[1:])

    def get_config(self) -> Dict:
        r"""
        Returns the configuration of the layer for serialization.

        Returns
        -------
        dict
            A dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"method": self.method, "threshold": self.threshold})
        return config


# For the ParallelReservoir model
@keras.saving.register_keras_serializable(package="MyLayers", name="InputSplitter")
class InputSplitter(keras.layers.Layer):
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
        ) + 1 > self.overlap, (
            "Overlap must be smaller than the length of each partition."
        )

        # Width of each partition including overlap
        partition_width = (features // self.partitions) + 2 * self.overlap

        # Circular wrapping
        wrapped_inputs = tf.concat(
            [inputs[..., -self.overlap :], inputs, inputs[..., : self.overlap]],
            axis=-1,
        )

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
