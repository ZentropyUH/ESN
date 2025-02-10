"""Custom Initializers."""

from typing import Dict, Union, List, Tuple

import networkx as nx
import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.sparse import linalg

from keras_reservoir_computing.utils.tf_utils import create_tf_rng

import keras
from keras import Initializer
from keras.src.initializers import RandomUniform


@keras.saving.register_keras_serializable(package="MyInitializers", name="InputMatrix")
class InputMatrix(Initializer):
    """
    Makes an initializer that generates an input matrix which connects the input to the reservoir.

    Every node in the network receives exactly one scalar input and
    every input is connected to N/D nodes. N being the number of nodes (columns)
    and D is the number of inputs (rows).
    Every non-zero element in the matrix is randomly generated from
    a uniform distribution in [-sigma,sigma]

    Args:
        sigma (float): Standard deviation of the uniform distribution.

    Returns:
        keras.initializers.Initializer: The initializer.

    #### Usage example:
    >>> w_init = InputMatrix(sigma=1)
    >>> w = tf.Variable(w_init((5,10)))
    >>> w is a 5x10 matrix with values in [-1, 1]
    """

    def __init__(
        self, sigma: float = 0.5, binarize: bool = False, seed: Union[int, None] = None
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

    def get_config(self) -> Dict:
        """Get the config dictionary of the initializer for serialization."""
        base_config = super().get_config()
        config = {"sigma": self.sigma, "ones": self.binarize, "seed": self.seed}

        config.update(base_config)
        return config
