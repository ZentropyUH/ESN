from typing import Dict, List, Union

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="SelectiveExponentiation")
class SelectiveExponentiation(tf.keras.layers.Layer):
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
        mask_f = tf.cast(
            tf.math.equal(tf.math.mod(tf.range(dim), 2), (self.index + 1) % 2),
            inputs.dtype,
        )

        # Elements to be exponentiated
        masked = inputs * mask_f
        # Elements to remain the same
        unmasked = inputs * (1.0 - mask_f)

        output = tf.math.pow(masked, self.exponent) + unmasked
        return output

    def compute_output_shape(self, input_shape: Union[tf.TensorShape, List[int]]) -> tf.TensorShape:
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
