from typing import Dict, List, Union

import tensorflow as tf


# For the ParallelReservoir model
@tf.keras.utils.register_keras_serializable(package="krc", name="FeaturePartitioner")
class FeaturePartitioner(tf.keras.layers.Layer):
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
