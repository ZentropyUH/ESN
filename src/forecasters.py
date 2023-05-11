"""Define the functions for the forecasting tasks."""
import numpy as np
from rich.progress import track

#### Forecasting ####


def classic_forecast(
    model,
    forecast_transient_data,
    val_data,
    val_target,
    forecast_length=100,
    save_name=None,
):
    """Forecast the given data.

    Args:
        model (keras.Model): The model to be used for the forecast.

        forecast_transient_data (np.array): The data to be used to initialize the forecast.

        val_data (np.array): The data to be forecasted.

        val_target (np.array): The target data.

        forecast_length (int, optional): The length of the forecast. Defaults to 100.

        callbacks (list, optional): The list of callback functions
            to be used during the forecast. Defaults to None.

        save_name (str, optional): The path to save the forecast. Defaults to None.
            If None, the forecast is not saved.

    Returns:
        (np.array, float): The predictions and the loss.
    """
    forecast_length = min(forecast_length, val_data.shape[1])
    val_target = val_target[:, :forecast_length, :]

    print()
    print(
        f"Forecasting free running sequence {forecast_length} steps ahead.\n"
    )

    print("    Ensuring ESP...\n")
    print("    Forecast transient data shape: ", forecast_transient_data.shape)
    model.predict(forecast_transient_data)

    # Initialize predictions with the first element of the validation data
    predictions = val_data[:, :1, :]

    print()
    print("    Predicting...\n")

    # Already tried initializing the predictions with shape (1, forecast_length, features)
    # and the performance was similar
    for _ in track(range(forecast_length)):
        pred = model(predictions[:, -1:, :])
        predictions = np.hstack((predictions, pred))

    # Eliminating the first element of the predictions
    predictions = predictions[:, 1:, :]
    print("    Predictions shape: ", predictions.shape)

    # Calculating the error
    try:
        loss = np.mean((predictions[0] - val_target[0]) ** 2)
    except ValueError:
        print("Error calculating the loss.")
        return np.inf

    print(f"Forecast loss: {loss}\n")

    if save_name is not None:
        print("Saving forecast...\n")
        # save the forecast in the forecasts folder
        np.save("".join([save_name, "_forecast.dt"]), predictions)

    return predictions


def section_forecast(
    model,
    forecast_transient_data,
    val_data,
    val_target,
    section_length,
    section_initialization_length,
    number_of_sections=1,
):
    """Forecast the given data in sections of length `section_length'.

    Everytime a section is forecasted, the model is reset back to zero and the last
    `section_initialization_length' elements corresponding to the val_target in the section
    are used to initialize the forecast of the next section.

    Args:
        model (keras.Model): The model to be used for the forecast.

        forecast_transient_data (np.array): The data to be used to initialize the whole forecast.

        val_data (np.array): The data to be forecasted.

        val_target (np.array): The target data.

        section_length (int): The length of each section.

        section_initialization_length (int): The length of the initialization of each section.

        number_of_sections (int, optional): The number of sections to be forecasted. Defaults to 1.

        callbacks (list, optional): The list of callback functions

    Returns:
        (np.array, dict): The predictions and the monitored variables.
    """
    # Initializing the predictions
    predictions = np.zeros((1, 0, val_data.shape[2]))

    for i in range(number_of_sections):
        print(f"Forecasting section {i+1} of {number_of_sections}.\n")
        forecast_section = classic_forecast(
            model,
            forecast_transient_data,
            val_data[:, i * section_length : (i + 1) * section_length, :],
            val_target[:, i * section_length : (i + 1) * section_length, :],
            forecast_length=section_length,
        )
        # Updating the predictions
        predictions = np.hstack((predictions, forecast_section))

        # Updating the forecast transient data
        forecast_transient_data = val_data[
            :,
            (i + 1) * section_length
            - section_initialization_length : (i + 1) * section_length,
            :,
        ]

    return predictions
