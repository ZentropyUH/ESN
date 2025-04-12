import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="SelectiveDropout")
class SelectiveDropout(tf.keras.layers.Layer):
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

    def __init__(self, mask: tf.Tensor, **kwargs) -> None:
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
        # Ensure inputs match expected dtype
        try:
            # Convert to float32 if necessary for consistent behavior
            inputs_float = tf.cast(inputs, tf.float32)
            # Apply selective dropout
            return tf.where(self.mask, 0.0, inputs_float)
        except tf.errors.InvalidArgumentError as e:
            raise ValueError(
                f"Failed to apply selective dropout: {e}. Check that mask shape {self.mask.shape} matches input feature dimension {inputs.shape[-1]}."
            )

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
