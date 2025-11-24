import os

import numpy as np
from natsort import natsorted
from scipy.stats import gmean

from .io import list_files_only, load_file


def compute_normalized_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the time-dependent normalized root mean square (RMS) error between predicted and true trajectories.

    Supports single and multiple trajectory comparisons. If `y_true` contains a single trajectory,
    all trajectories in `y_pred` are compared to it. If `y_true` contains multiple trajectories,
    `y_pred` must have the same shape, and comparisons are made pairwise.

    Parameters
    ----------
    y_true : np.ndarray
        Reference trajectories with shape:
        - (time, features): A single reference trajectory. All `y_pred` samples are compared against this.
        - (samples, time, features): Multiple reference trajectories, where each `y_true[i]` is compared to `y_pred[i]`.

    y_pred : np.ndarray
        Predicted trajectories with shape:
        - (samples, time, features) if `y_true` contains multiple trajectories.
        - (time, features) or (samples, time, features) if `y_true` contains a single trajectory.

    Returns
    -------
    np.ndarray
        A 1D array of shape `(time,)`, representing the normalized RMS error at each time step,
        averaged over all samples when applicable.

    Raises
    ------
    ValueError
        If `y_true` and `y_pred` have incompatible shapes.

    Notes
    -----
    - The normalization scale is computed per sample using the RMS of `y_true[i]` over features.
    - Errors are computed as the geometric mean across samples at each time step.

    Example
    -------
    >>> import numpy as np
    >>> y_true = np.array([
    ...     [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # Sample 1
    ...     [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]   # Sample 2
    ... ])
    >>> y_pred = np.array([
    ...     [[1.1, 2.1], [1.9, 3.1], [3.1, 3.9]],  # Slightly deviating predictions
    ...     [[1.3, 2.4], [2.8, 3.4], [3.4, 4.6]]
    ... ])
    >>> compute_normalized_error(y_true, y_pred)
    array([0.043, 0.051, 0.034])    # Example output, values will vary

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[None, :, :]

    N_pred = y_pred.shape[0]

    if y_true.ndim == 2 or y_true.shape[0] == 1:
        y_true = np.tile(y_true.reshape(1, *y_true.shape[-2:]), (N_pred, 1, 1))

    if y_pred.shape != y_true.shape:
        raise ValueError(
            "When y_true has shape (samples, time, features), y_pred must match that shape."
        )

    # Compute reference scale per sample
    ref_scale = np.linalg.norm(np.sqrt(np.mean(y_true**2, axis=1)), axis=1, keepdims=True)

    # Compute error norms
    diff = np.linalg.norm(y_pred - y_true, axis=2)

    # Normalize
    norm_error = diff / ref_scale  # Shape (samples, time)

    # Compute geometric mean across samples
    final_error = gmean(norm_error, axis=0)

    return final_error


def mean_ensemble_prediction(predictions: np.ndarray) -> np.ndarray:
    """
    Compute the mean ensemble prediction across multiple models.

    The input consists of multiple models predicting multiple samples over time and features.
    This function averages over the model axis, producing a mean prediction per sample.

    Parameters
    ----------
    predictions : np.ndarray
        Array of shape `(models, samples, time, features)`, containing predictions from multiple models.

    Returns
    -------
    np.ndarray
        Mean prediction of shape `(samples, time, features)`, averaged over all models.
    """

    return np.mean(predictions, axis=0)


def get_all_predictions(predictions_path: str) -> np.ndarray:
    """
    Load all prediction files from a directory and return them as a NumPy array.

    Prediction files must end with `_predictions.[format]`, where `[format]` can be `npy`, `csv`, or `nc`.

    Parameters
    ----------
    predictions_path : str
        Path to the directory containing the prediction files.

    Returns
    -------
    np.ndarray
        Loaded predictions with shape `(models, samples, time, features)`.

    Raises
    ------
    ValueError
        If no prediction files are found in the directory.
    """

    # Get all files in the directory
    files = list_files_only(predictions_path)

    # Filter out the prediction files
    prediction_files = [f for f in files if "_predictions" in f]
    prediction_files = natsorted(prediction_files)

    # Load the first prediction to determine the shape
    first_prediction = load_file(os.path.join(predictions_path, prediction_files[0]))
    predictions_shape = (len(prediction_files),) + first_prediction.shape

    # Initialize the predictions array
    predictions = np.empty(predictions_shape, dtype=first_prediction.dtype)

    # Load the predictions into the array
    for i, file in enumerate(prediction_files):
        predictions[i] = load_file(os.path.join(predictions_path, file))

    return predictions


def get_all_targets(predictions_path: str) -> np.ndarray:
    """
    Load all target files from a directory and return them as a NumPy array.

    Target files must end with `_targets.[format]`, where `[format]` can be `npy`, `csv`, or `nc`.

    Parameters
    ----------
    predictions_path : str
        Path to the directory containing the target files.

    Returns
    -------
    np.ndarray
        Loaded target data with shape `(samples, time, features)`.

    Raises
    ------
    ValueError
        If no target files are found in the directory.
    """

    # Get all files in the directory
    files = list_files_only(predictions_path)

    # Filter out the target files
    target_files = [f for f in files if "_targets" in f]
    target_files = natsorted(target_files)

    # Load the first target to determine the shape
    first_target = load_file(os.path.join(predictions_path, target_files[0]))
    targets_shape = (len(target_files),) + first_target.shape

    # Initialize the targets array
    targets = np.empty(targets_shape, dtype=first_target.dtype)

    # Load the targets into the array
    for i, file in enumerate(target_files):
        targets[i] = load_file(os.path.join(predictions_path, file))

    return targets


def get_all_errors(predictions_path: str) -> np.ndarray:
    """
    Compute the normalized error for all prediction-target pairs in a directory.

    This function loads all predictions and targets from a given directory,
    computes the normalized error for each sample, and returns the results.

    Parameters
    ----------
    predictions_path : str
        Path to the directory containing the predictions and targets.

    Returns
    -------
    np.ndarray
        Normalized errors of shape `(samples, time, 1)`.
    """

    preds = get_all_predictions(predictions_path)
    targets = get_all_targets(predictions_path)

    errors = np.empty((preds.shape[0], preds.shape[2], 1))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        errors[i, :, 0] = compute_normalized_error(target, pred)

    return errors


def mean_prediction_error(predictions_path: str) -> np.ndarray:
    """
    Compute the normalized error of the mean ensemble prediction.

    This function:
    1. Loads all predictions.
    2. Computes the mean prediction over all models.
    3. Loads the target data.
    4. Computes the normalized error between the mean prediction and targets.

    Parameters
    ----------
    predictions_path : str
        Path to the directory containing the predictions.

    Returns
    -------
    np.ndarray
        Normalized error of shape `(samples, time)`.
    """

    preds = get_all_predictions(predictions_path)
    mean_pred = mean_ensemble_prediction(preds)
    targets = get_all_targets(predictions_path)

    error = compute_normalized_error(targets, mean_pred)

    return error
