from math import gcd
from typing import Optional, Union

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="krc", name="BinaryBalancedInitializer"
)
class BinaryBalancedInitializer(tf.keras.Initializer):
    r"""
    Deterministic initializer that generates a **dense** input matrix with entries in ``{-1, +1}``,
    column-wise **balance** (as close to zero as possible), and **low inter-column correlation**
    via truncated Walsh–Hadamard structure.

    The matrix is built **without randomness** and is suitable for ESN/RC setups where each
    column (i.e., each reservoir unit) should receive a balanced mix of positive/negative signs
    from the input channels, and columns should be nearly orthogonal.

    Construction (shape = (rows, cols) = (inputs, units))
    -----------------------------------------------------
    1) Let ``n_work = rows`` if ``rows`` is even, else ``n_work = rows + 1``.
    2) Build a Sylvester Hadamard matrix ``H_L`` with ``L = 2^ceil(log2(max(n_work, cols+1)))``.
       Exclude the DC (all-ones) column.
    3) Select ``cols`` columns deterministically with a constant step ``s`` that is coprime with ``L``:
       indices: ``1 + (j * s) % (L-1)`` for ``j = 0..cols-1``.
    4) Truncate to the first ``n_work`` rows, then **balance each column** to make its sum zero
       (possible because ``n_work`` is even) by minimal sign flips (bottom-to-top).
    5) If ``rows`` is odd, delete one row (deterministically the row minimizing the absolute row-sum),
       so each column sums to ``±1`` (the optimal imbalance for odd cardinality). Optionally
       **globally balance** the signs across columns by flipping whole columns so that counts
       of ``+1``/``-1`` column-sums differ by at most one.

    Parameters
    ----------
    input_scaling : float, optional
        If provided, scales the resulting dense matrix by this factor.
    balance_global : bool, default=True
        When ``rows`` is odd, enforce a near-equal global count of ``+1`` vs ``-1`` column-sums
        by flipping full columns (does not change correlation magnitudes).
    step : int, optional
        Preferred column selection step. If ``None`` or not coprime with ``L``, the initializer
        will choose the smallest odd step that is coprime with ``L``.
    seed : int, tf.random.Generator, or None, default=None
        Unused (kept for API symmetry). The initializer is fully deterministic regardless of seed.

    Returns
    -------
    tf.Tensor
        A dense tensor of shape ``(rows, cols)`` with values in ``{-1, +1}`` (or scaled if
        ``input_scaling`` is provided).

    Notes
    -----
    - Column-wise balance: sum over rows per column is exactly ``0`` if ``rows`` even, else ``±1``.
    - Low correlation: columns originate from orthogonal Hadamard columns before truncation/balancing.
    - Deterministic: no RNG is used; the output is reproducible given the shape and parameters.

    Examples
    --------
    >>> init = DeterministicBinaryBalancedInitializer(input_scaling=0.5, balance_global=True)
    >>> W_in = init((3, 10))  # 3 inputs × 10 units
    >>> tf.reduce_sum(W_in, axis=0)  # each column sums to ±1 (rows=3 is odd)
    """

    def __init__(
        self,
        input_scaling: Optional[float] = None,
        balance_global: bool = True,
        step: Optional[int] = None,
        seed: Union[int, tf.random.Generator, None] = None,
    ) -> None:
        """Initialize the initializer."""
        self.input_scaling = input_scaling
        self.balance_global = balance_global
        self.step = step
        self.seed = seed  # kept for API symmetry; not used

    # ---------- Internal helpers (NumPy) ----------

    @staticmethod
    def _next_pow2(x: int) -> int:
        """Return the next power of two >= x."""
        return 1 << (x - 1).bit_length()

    @staticmethod
    def _hadamard(L: int) -> np.ndarray:
        """Construct Sylvester Hadamard matrix H_L ∈ {+1, -1}^{L×L}, with L a power of 2."""
        H = np.array([[1]], dtype=np.int8)
        while H.shape[0] < L:
            H = np.block([[H,  H],
                          [H, -H]]).astype(np.int8)
        return H

    @staticmethod
    def _choose_step(L: int, preferred: Optional[int]) -> int:
        """
        Choose a step s coprime with L for column selection. If `preferred` is valid, use it;
        otherwise pick the smallest odd s ≥ 1 with gcd(s, L) == 1.
        """
        if preferred is not None and gcd(preferred, L) == 1:
            return int(preferred)
        s = 1
        # Ensure coprime with L (L is power-of-two, so any odd s works).
        while gcd(s, L) != 1:
            s += 2
        return s

    @staticmethod
    def _balance_columns_zero_sum(Vw: np.ndarray) -> None:
        """
        Make each column sum to 0 by minimal sign flips, assuming Vw has even number of rows.
        Operates in-place.
        """
        n_work, m = Vw.shape
        for j in range(m):
            s_j = int(Vw[:, j].sum())
            if s_j == 0:
                continue
            # Flip signs bottom-to-top to correct in ±2 increments
            for i in range(n_work - 1, -1, -1):
                if s_j == 0:
                    break
                if s_j > 0 and Vw[i, j] == 1:
                    Vw[i, j] = -1
                    s_j -= 2
                elif s_j < 0 and Vw[i, j] == -1:
                    Vw[i, j] = 1
                    s_j += 2

    @staticmethod
    def _delete_least_bias_row(Vw: np.ndarray) -> np.ndarray:
        """
        Delete one row (deterministically) to minimize overall bias when rows is odd.
        Returns a new array with one fewer row.
        """
        row_sums = Vw.sum(axis=1)
        r_del = int(np.argmin(np.abs(row_sums)))
        return np.delete(Vw, r_del, axis=0)

    @staticmethod
    def _balance_global_column_counts(V: np.ndarray) -> None:
        """
        For odd number of rows, each column sum is ±1. Make the global counts of +1 and -1
        column-sums as equal as possible by flipping entire columns. Operates in-place.
        """
        m = V.shape[1]
        col_sums = V.sum(axis=0)  # each is ±1
        total = int(col_sums.sum())
        target = 0 if (m % 2 == 0) else 1  # desired net imbalance

        if total > target:
            flips = (total - target) // 2  # each column flip reduces total by 2
            cnt = 0
            for j in range(m):
                if col_sums[j] == 1 and cnt < flips:
                    V[:, j] *= -1
                    col_sums[j] = -1
                    cnt += 1
        elif total < -target:
            flips = (-target - total) // 2
            cnt = 0
            for j in range(m):
                if col_sums[j] == -1 and cnt < flips:
                    V[:, j] *= -1
                    col_sums[j] = 1
                    cnt += 1

    # ---------- Keras initializer API ----------

    def __call__(self, shape: tuple, dtype=tf.float32) -> tf.Tensor:
        """
        Generate the matrix of the given shape.

        Parameters
        ----------
        shape : tuple
            Target shape ``(rows, cols) = (inputs, units)``.
        dtype : tf.dtypes.DType
            Output dtype.

        Returns
        -------
        tf.Tensor
            Dense matrix with entries in ``{-1, +1}`` (scaled if `input_scaling` is set).
        """
        dims = tf.TensorShape(shape).as_list()  # -> list[int|None]

        if dims is None:
            raise ValueError("Rank of shape unknown at initialization time.")
        if len(dims) == 1:
            rows, cols = int(dims[0]), 1
        elif len(dims) == 2:
            rows, cols = map(int, dims)
        else:
            raise ValueError(f"Shape must be 1D or 2D, got {shape}.")

        if rows == 0 or cols == 0:
            return tf.zeros((rows, cols), dtype=dtype)

        # 1) Work with even row count
        n_work = rows if (rows % 2 == 0) else rows + 1

        # 2) Build H_L and select columns deterministically (avoid DC column 0)
        L = self._next_pow2(max(n_work, cols + 1, 2))
        H = self._hadamard(L)  # (L, L) in {±1}

        s = self._choose_step(L, self.step)
        idxs = [1 + (j * s) % (L - 1) for j in range(cols)]  # exclude index 0 (DC)

        Vw = H[:n_work, idxs].copy()  # shape (n_work, cols)

        # 3) Balance each column to sum zero (n_work is even)
        self._balance_columns_zero_sum(Vw)

        # 4) If rows is even, done; else delete one row and optionally balance global counts
        if n_work == rows:
            V = Vw
        else:
            V = self._delete_least_bias_row(Vw)
            if self.balance_global:
                self._balance_global_column_counts(V)

        V = V.astype(np.float32)

        if self.input_scaling is not None:
            V *= float(self.input_scaling)

        return tf.convert_to_tensor(V, dtype=dtype)

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        base = super().get_config()
        base.update(
            {
                "input_scaling": self.input_scaling,
                "balance_global": self.balance_global,
                "step": self.step,
                "seed": self.seed,
            }
        )
        return base
