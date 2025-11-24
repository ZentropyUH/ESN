from typing import Union
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="RingWindowInputInitializer")
class RingWindowInputInitializer(tf.keras.Initializer):
    """
    Deterministic input-initializer that feeds each input channel into a
    contiguous window on the *core ring* of a dendrocycle(+chords) reservoir.

    Mapping: rows = input channels (m), cols = reservoir nodes (n).
    Only the first C = round(c * n) columns (core) receive nonzeros.

    Parameters
    ----------
    c : float
        Fraction of core ring nodes. First round(c*n) columns are core.
    window : int | float
        If int >= 1: number of core nodes per channel window.
        If float in (0, 1]: fraction of C per channel window (rounded >= 1).
    taper : {"flat","triangle","cosine"}, default "flat"
        Weight profile *within a channel's window* centered on its center index.
        - flat     : all ones
        - triangle : linear to 0 at the edges (peak 1 at center)
        - cosine   : Hann-like toward 0 at the edges (peak 1 at center)
    signed : {"allpos","alt_ring","alt_inputs"}, default "allpos"
        Sign policy:
        - allpos    : positive weights
        - alt_ring  : alternate sign by ring column index (…,+,-,+,-,…)
        - alt_inputs: alternate sign by input row (row k gets (-1)^k)
    gain : float, default 1.0
        Per-channel L2-norm after taper/sign are applied.

    Notes
    -----
    - No randomness. Fully reproducible.
    - Dendrites/quiescent columns (j >= C) are zero.
    """

    def __init__(
        self,
        c: float,
        window: Union[int, float],
        taper: str = "flat",  # {"flat","triangle","cosine"}
        signed: str = "allpos",  # {"allpos","alt_ring","alt_inputs"}
        gain: float = 1.0,
    ) -> None:
        if not (0 < c <= 1):
            raise ValueError("c must be in (0,1].")
        self.c = float(c)

        if isinstance(window, int):
            if window < 1:
                raise ValueError("window int must be >= 1.")
            self.window = window
            self.window_is_frac = False
        elif isinstance(window, float):
            if not (0 < window <= 1):
                raise ValueError("window float must be in (0,1].")
            self.window = float(window)
            self.window_is_frac = True
        else:
            raise TypeError("window must be int or float.")

        if taper not in {"flat", "triangle", "cosine"}:
            raise ValueError("taper must be 'flat', 'triangle', or 'cosine'.")
        if signed not in {"allpos", "alt_ring", "alt_inputs"}:
            raise ValueError("signed must be 'allpos', 'alt_ring', or 'alt_inputs'.")
        if not (np.isfinite(gain) and gain > 0):
            raise ValueError("gain must be a positive finite float.")

        self.taper = taper
        self.signed = signed
        self.gain = float(gain)

    def _window_size(self, C: int) -> int:
        if self.window_is_frac:
            W = int(round(self.window * C))
        else:
            W = int(self.window)
        return max(1, min(C, W))

    @staticmethod
    def _taper_vector(W: int, kind: str) -> np.ndarray:
        # Ensure nonzero for tiny windows
        if W <= 2:
            return np.ones(W, dtype=np.float64)

        idx = np.arange(W, dtype=np.float64)
        if kind == "flat":
            w = np.ones(W, dtype=np.float64)
        elif kind == "triangle":
            center = (W - 1) / 2.0
            w = 1.0 - np.abs(idx - center) / center
        elif kind == "cosine":
            center = (W - 1) / 2.0
            x = (idx - center) / center  # [-1,1]
            w = 0.5 * (1.0 + np.cos(np.pi * x))  # 0 at edges, 1 at center
        else:
            raise RuntimeError("unreachable")
        return np.clip(w, 0.0, None)

    def __call__(self, shape, dtype=None):
        if not (isinstance(shape, (tuple, list)) and len(shape) == 2):
            raise ValueError(f"shape must be (m, n); got {shape!r}")
        m, n = int(shape[0]), int(shape[1])
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive.")

        tf_dtype = tf.as_dtype(dtype or tf.float32)
        np_dtype = tf_dtype.as_numpy_dtype

        C = max(1, int(round(self.c * n)))  # core size
        W = self._window_size(C)

        B = np.zeros((m, n), dtype=np_dtype)
        base = self._taper_vector(W, self.taper).astype(np_dtype)

        def window_indices(center: int) -> np.ndarray:
            start = center - (W // 2)
            return (np.arange(start, start + W) % C).astype(int)

        for k in range(m):
            start = int(np.floor(k * C / m)) % C
            center = (start + (W - 1) // 2) % C

            cols_core = window_indices(center)
            row_vals = base.copy()

            if self.signed == "allpos":
                signs = 1.0
            elif self.signed == "alt_ring":
                signs = (1.0 - 2.0 * (cols_core % 2)).astype(np_dtype)
            else:  # "alt_inputs"
                signs = 1.0 if (k % 2 == 0) else -1.0

            row_vals = row_vals * signs
            norm = float(np.linalg.norm(row_vals))
            scaled = row_vals if norm == 0.0 else (self.gain / norm) * row_vals

            B[k, cols_core] = scaled  # only core; rest stay zero

        return tf.convert_to_tensor(B, dtype=tf_dtype)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "c": self.c,
                "window": self.window,
                "taper": self.taper,
                "signed": self.signed,
                "gain": self.gain,
            }
        )
        return base
