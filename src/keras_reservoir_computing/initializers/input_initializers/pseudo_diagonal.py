"""Custom Initializers."""

from typing import Optional, Union

import tensorflow as tf

from keras_reservoir_computing.utils.tensorflow import create_tf_rng


@tf.keras.utils.register_keras_serializable(
    package="krc", name="PseudoDiagonalInitializer"
)
class PseudoDiagonalInitializer(tf.keras.Initializer):
    """
    An initializer that generates an input matrix connecting inputs to reservoir nodes.

    Each node in the reservoir receives exactly one scalar input, and each input is connected
    to approximately `N/D` nodes, where `N` is the number of nodes (columns) and `D` is the
    number of inputs (rows). The non-zero elements of the matrix are sampled from a uniform
    distribution in the range `[-sigma, sigma]`.

    Parameters
    ----------
    sigma : float, optional
        Input scaling factor. If None, the rescaling is disabled.
    binarize : bool, default=False
        If True, the matrix values are binarized to `{-sigma, sigma}`.
    seed : int, tf.random.Generator, or None, default=None
        Seed for random number generation.

    Returns
    -------
    tf.Tensor
        The initialized weight matrix matching the requested shape.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import PseudoDiagonalInitializer
    >>> w_init = PseudoDiagonalInitializer(sigma=1, binarize=True, seed=42)
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix with values in [-1, 1], two non-zero values per row.
    """

    def __init__(
        self,
        input_scaling: Optional[float] = None,
        binarize: bool = False,
        seed: Union[int, tf.random.Generator, None] = None,
    ) -> None:
        """Initialize the initializer."""

        self.input_scaling = input_scaling
        self.binarize = binarize
        self.seed = seed

        self.tf_rng = create_tf_rng(seed)

    def __call__(
        self,
        shape: tuple,
        dtype=tf.float32,
    ) -> tf.Tensor:
        """
        Generate the matrix of the given shape.

        Parameters
        ----------
        shape :
            Shape of the matrix, (m, n). If an integer, creates an m x m matrix.
        dtype : tf.dtypes.DType
            Data type of the matrix.

        Returns
        -------
        tf.Tensor
            The block-diagonal style matrix.
        """

        dims = tf.TensorShape(shape).as_list()  # -> list[int|None]

        if dims is None:
            raise ValueError("Rank of shape unknown at initialization time.")
        if len(dims) == 1:
            rows, cols = int(dims[0]), 1
        elif len(dims) == 2:
            rows, cols = map(int, dims)
        else:
            raise ValueError(f"Shape must be 1D or 2D, got {shape}")
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
        values = self.tf_rng.uniform((num_values,), minval=-1, maxval=1, dtype=dtype)
        if self.binarize:
            values = tf.sign(values)

        # Construct the sparse tensor
        w_in = tf.SparseTensor(
            indices=tf.cast(indices_nonzero, tf.int64),
            values=values,
            dense_shape=[rows, cols],
        )
        w_in = tf.sparse.reorder(w_in)
        w_in = tf.sparse.to_dense(w_in)

        if self.input_scaling is not None:
            w_in *= self.input_scaling

        return w_in

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {
            "input_scaling": self.input_scaling,
            "binarize": self.binarize,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
