from typing import List, Union

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="OutliersFilteredMean")
class OutliersFilteredMean(tf.keras.layers.Layer):
    r"""
    A Keras layer that removes outliers (along the `samples` dimension) independently at
    each (batch, timestep) location, based on a specified method (Z-score or IQR), and
    then returns the mean of the remaining elements.

    This layer is useful for denoising temporal data by filtering out extreme values.

    **Input Shape**:
        ``(samples, batch, timesteps, features)``

    **Output Shape**:
        ``(batch, timesteps, features)``

    **Procedure**:
        1. Compute the L2 norm over the last dimension (features), resulting in shape
           ``(samples, batch, timesteps)``.
        2. For each `(batch, timestep)`, compute outlier thresholds using:
           - **Z-score method**: mean and std deviation across `samples`.
           - **IQR method**: first and third quartiles across `samples`.
        3. Build a mask indicating inlier vs. outlier samples.
        4. Compute the mean of the inlier samples.
        5. If an entire `(batch, timestep)` has no valid samples, fallback to numerical stability.

    Parameters
    ----------
    method : str, optional
        Outlier removal method. Choices: ``{"z_score", "iqr"}``. Defaults to ``"z_score"``.
    threshold : float, optional
        Threshold for removing outliers (e.g., 3.0 for ±3 std if using Z-score). Defaults to 3.0.
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    method : str
        The chosen outlier detection method.
    threshold : float
        The threshold value used for filtering.

    Raises
    ------
    ValueError
        If `method` is not one of ``"z_score"`` or ``"iqr"``.
        If using IQR method, TensorFlow Probability (`tfp`) must be installed.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.custom_layers import OutliersFilteredMean
    >>> layer = OutliersFilteredMean(method="z_score", threshold=2.0)
    >>> x = tf.random.normal((10, 3, 5, 4))  # 10 samples, batch=3, timesteps=5, features=4
    >>> y = layer([x,x,x])
    >>> print(y.shape)
    (3, 5, 4)
    """

    def __init__(self, method: str = "z_score", threshold: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold

        if self.method not in ["z_score", "iqr"]:
            raise ValueError(f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'.")

    def build(self, input_shape):
        # No trainable parameters to build
        super().build(input_shape)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        Removes outliers at each (batch, timestep), then averages over the samples dimension.

        Parameters
        ----------
        inputs : Union[tf.Tensor, List[tf.Tensor]]
            A tensor of shape ``(samples, batch, timesteps, features)``.

        Returns
        -------
        tf.Tensor
            A tensor of shape ``(batch, timesteps, features)``, representing the mean of the
            non-outlier samples.
        """
        if isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=0)  # (samples, batch, None, features)
        else:
            inputs = tf.expand_dims(inputs, axis=0)  # (1, batch, None, features)

        # 1) Compute the norm over the last dimension => shape (samples, batch, timesteps)
        norms = tf.norm(inputs, axis=-1)

        # 2) For each (batch, timestep), figure out which samples are inliers vs outliers
        if self.method == "z_score":
            # mean_norm, std_norm => shape (batch, timesteps)
            mean_norm = tf.reduce_mean(norms, axis=0)
            std_norm = tf.math.reduce_std(norms, axis=0)

            # Avoid division by zero by forcing std_norm to 1 where it's 0
            std_norm = tf.where(std_norm > 0, std_norm, tf.ones_like(std_norm))

            # z-scores => shape (samples, batch, timesteps)
            z_scores = tf.abs((norms - mean_norm) / std_norm)

            # True => sample is inlier, False => outlier
            mask = z_scores < self.threshold

        else:  # self.method == "iqr"
            # Q1, Q3 => shape (batch, timesteps)
            q1 = tf.keras.ops.quantile(norms, 0.25, axis=0)
            q3 = tf.keras.ops.quantile(norms, 0.75, axis=0)
            iqr = q3 - q1

            # Lower/upper bounds
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr

            # True => inlier, False => outlier
            mask = (norms >= lower_bound) & (norms <= upper_bound)

        # mask shape => (samples, batch, timesteps)
        # We need this mask to broadcast along the features dimension for the final averaging.
        mask_expanded = tf.cast(tf.expand_dims(mask, axis=-1), dtype=inputs.dtype)
        # mask_expanded => (samples, batch, timesteps, 1)

        # 3) Multiply inputs by mask to zero out outliers
        masked_inputs = inputs * mask_expanded
        # shape => (samples, batch, timesteps, features)

        # 4) Sum along the samples dimension => shape (batch, timesteps, features)
        sum_ = tf.reduce_sum(masked_inputs, axis=0)

        # 5) Count inlier samples at each (batch, timestep) => shape (batch, timesteps, 1)
        count_ = tf.reduce_sum(mask_expanded, axis=0, keepdims=False)
        count_ = tf.broadcast_to(count_, tf.shape(sum_))  # (batch, timesteps, features)

        # Avoid dividing by zero: if count_ is 0, replace with 1
        count_ = tf.where(count_ > 0, count_, tf.ones_like(count_))

        # 6) Take the mean => shape (batch, timesteps, features)
        mean_ = sum_ / count_

        return mean_

    def compute_output_shape(self, input_shape):
        # If input is a list, use one element’s shape (they all have the same shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]  # Single tensor shape: (batch, timesteps, features)

        # Ensure it's the expected format (batch, timesteps, features) before returning
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]])

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method, "threshold": self.threshold})
        return config
