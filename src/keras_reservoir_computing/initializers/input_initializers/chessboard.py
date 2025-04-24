from typing import Optional

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="krc", name="ChessboardInitializer"
)
class ChessboardInitializer(tf.keras.Initializer):
    """
    An initializer that generates a chessboard pattern with values in {-1, 1}.

    Parameters
    ----------
    input_scaling : float, optional
        The scaling factor for the input values. Default is 1.

    Returns
    -------
    tf.Tensor
        The initialized weight matrix matching the requested shape.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import ChessboardInitializer
    >>> w_init = ChessboardInitializer()
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix with values in {-1, 1}.
    """

    def __init__(self, input_scaling: Optional[float] = None) -> None:
        """Initialize the initializer."""
        self.input_scaling = input_scaling
        super().__init__()

    def __call__(self, shape: tuple, dtype: Optional[tf.DType] = None) -> tf.Tensor:
        i = np.arange(shape[0])[:, None]
        j = np.arange(shape[1])[None, :]
        W = (-1) ** (i + j)
        diag_indices = np.diag_indices(min(shape))
        W[diag_indices] = (-1) ** np.arange(min(shape))

        if self.input_scaling is not None:
            W *= self.input_scaling

        W = W.astype(dtype.as_numpy_dtype)

        return tf.convert_to_tensor(W, dtype=dtype)


    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        base_config = super().get_config()
        return base_config
