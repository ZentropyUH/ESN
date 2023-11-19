import os
import sys
import pandas as pd
from functions import forecast_from_saved_model
from LZ.main import generate_plots

def forecast_and_save(model_path, data_folder, output_folder, forecast_length):

    #  Ver si las carpetas de datos y de salida existen
    if not os.path.exists(data_folder):
        raise ValueError(f"The folder does not exist: {data_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterar por los archivos en la carpeta de datos
    for file in os.listdir(data_folder):
        if file .endswith('.csv'):  # Asegúrate de que es un archivo CSV
            data_path = os.path.join(data_folder, file)

            output_path = os.path.join(output_folder, file)

            print(f"Processing file {file}")
            
            forecast_from_saved_model(trained_model_path=model_path, 
                                    data_file=data_path,
                                    output_dir=output_path,
                                    forecast_method="classic",
                                    forecast_length=forecast_length,
                                    section_initialization_length=50,
                                    number_of_sections=10,
                                    internal_states=True,
                                    feedback_metrics=False
                                    )

            print("Forecast end for all files")
            
    return output_folder


if __name__ == "__main__":
    output = forecast_and_save(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    generate_plots(output, "/data/tsa/destevez/sheyls/DATA/ARTIFICIAL/Lorenz")