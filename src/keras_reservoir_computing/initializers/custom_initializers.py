"""Custom Initializers."""

from typing import List, Tuple, Union

import keras
import tensorflow as tf
from keras import Initializer

from keras_reservoir_computing.utils.tf_utils import create_tf_rng


@keras.saving.register_keras_serializable(package="krc", name="InputMatrix")
class InputMatrix(Initializer):
    """
    An initializer that generates an input matrix connecting inputs to reservoir nodes.

    Each node in the reservoir receives exactly one scalar input, and each input is connected
    to approximately `N/D` nodes, where `N` is the number of nodes (columns) and `D` is the
    number of inputs (rows). The non-zero elements of the matrix are sampled from a uniform
    distribution in the range `[-sigma, sigma]`.

    Parameters
    ----------
    sigma : float, default=0.5
        The maximum absolute value of the uniform distribution.
    binarize : bool, default=False
        If True, the matrix values are binarized to `{-sigma, sigma}`.
    seed : int, tf.random.Generator, or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    sigma : float
        The maximum absolute value for the uniform distribution.
    binarize : bool
        Whether to binarize the matrix values.
    seed : int, tf.random.Generator, or None
        Seed for random number generation.
    tf_rng : tf.random.Generator
        TensorFlow random generator initialized with the given seed.

    Methods
    -------
    __call__(shape, dtype=tf.float32)
        Generates the initialized input matrix with the given shape.
    get_config()
        Returns a dictionary containing the initializer's configuration.

    Returns
    -------
    keras.initializers.Initializer
        The initializer instance.

    Examples
    --------
    >>> w_init = InputMatrix(sigma=1, binarize=True, seed=42)
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix with values in [-1, 1], two non-zero values per row.
    """
    def __init__(
        self, sigma: float = 0.5, binarize: bool = False, seed: Union[int, tf.random.Generator, None] = None
    ) -> None:
        """Initialize the initializer."""
        assert sigma > 0, "sigma must be positive"

        self.sigma = sigma
        self.binarize = binarize
        self.seed = seed

        self.tf_rng = create_tf_rng(seed)

    def __call__(
        self,
        shape: Union[int, Tuple[int, int], List[int]],
        dtype=tf.float32,
    ) -> tf.Tensor:
        """
        Generate the matrix of the given shape.

        Parameters
        ----------
        shape : int or tuple of two ints
            Shape of the matrix, (m, n). If an integer, creates an m x m matrix.
        dtype : tf.dtypes.DType
            Data type of the matrix.

        Returns
        -------
        tf.Tensor
            The block-diagonal style matrix.
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        elif not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            raise ValueError(
                "Shape must be an integer or a tuple/list of two integers."
            )

        rows, cols = shape
        # We will build up indices and values for a tf.SparseTensor

        if rows == 0 or cols == 0:
            # Edge case: trivial shape
            return tf.zeros(shape, dtype=dtype)

        # Case 1: rows <= cols
        if rows <= cols:
            block_size = cols // rows
            remainder = cols % rows
            # For row r, assign columns in a contiguous block [start, end)
            # start = r * block_size + min(r, remainder)
            # end   = start + block_size + (1 if r < remainder else 0)

            indices_list = []
            for r in range(rows):
                start_col = r * block_size + min(r, remainder)
                end_col = start_col + block_size + (1 if r < remainder else 0)
                # Build (row, col) pairs
                row_coords = tf.fill([end_col - start_col], tf.cast(r, tf.int32))
                col_coords = tf.range(start_col, end_col, dtype=tf.int32)
                r_stack = tf.stack([row_coords, col_coords], axis=1)
                indices_list.append(r_stack)

            indices_nonzero = tf.concat(indices_list, axis=0)

        # Case 2: rows > cols
        else:
            block_size = rows // cols
            remainder = rows % cols
            # For column c, assign rows in a contiguous block [start, end)

            indices_list = []
            for c in range(cols):
                start_row = c * block_size + min(c, remainder)
                end_row = start_row + block_size + (1 if c < remainder else 0)
                # Build (row, col) pairs
                row_coords = tf.range(start_row, end_row, dtype=tf.int32)
                col_coords = tf.fill([end_row - start_row], tf.cast(c, tf.int32))
                c_stack = tf.stack([row_coords, col_coords], axis=1)
                indices_list.append(c_stack)

            indices_nonzero = tf.concat(indices_list, axis=0)

        # Number of non-zero entries
        num_values = tf.shape(indices_nonzero)[0]

        # Sample values in [-sigma, sigma]
        values = self.tf_rng.uniform(
            (num_values,), minval=-self.sigma, maxval=self.sigma, dtype=dtype
        )
        if self.binarize:
            values = tf.sign(values) * self.sigma

        # Construct the sparse tensor
        w_in = tf.SparseTensor(
            indices=tf.cast(indices_nonzero, tf.int64),
            values=values,
            dense_shape=[rows, cols],
        )
        w_in = tf.sparse.reorder(w_in)
        return tf.sparse.to_dense(w_in)

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.
        
        Returns
        -------
        dict
            The configuration dictionary.
        """
        base_config = super().get_config()
        config = {"sigma": self.sigma, "binarize": self.binarize, "seed": self.seed}

        config.update(base_config)
        return config
