import os
import shutil
import sys

def group_dimensions_for_plot(main_folder_path, destiny_folder):
    # Verificar las subcarpetas dentro de la carpeta principal
    subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]
    i = 0

    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            if file_name.endswith(".json"):
                i += 1
                # Obtiene el nombre del archivo sin extensión
                base_name = file_name.split('_')[-1].split('.')[0]
                # Ruta de destino
                destiny_folder_path = os.path.join(destiny_folder, f"dimension_{base_name}")
                # Crear la carpeta de destino si no existe
                os.makedirs(destiny_folder_path, exist_ok=True)
                # Mover el archivo
                #TODO
                os.system(f"mv {subfolder}/" + file_name + " " + f"{destiny_folder_path}/" + file_name.replace(".txt", str(i)))                


def transpose_csv_files(data_internal_state, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    # Obtener una lista de todos los archivos CSV en la carpeta
    csv_files = [f for f in os.listdir(data_internal_state) if f.endswith('.csv')]

    # Transponer cada archivo CSV y guardar el resultado
    for csv_file in csv_files:
        # Cargar el archivo CSV
        df = pd.read_csv(os.path.join(data_internal_state, csv_file))

        # Transponer el DataFrame
        df_transposed = df.transpose().reset_index(drop=True)


        # Construir el nombre del archivo de salida
        output_file_name = os.path.splitext(csv_file)[0] + '_transposed.csv'

        # Guardar el DataFrame transpuesto en un nuevo archivo CSV
        df_transposed.to_csv(os.path.join(output_folder_path, output_file_name), header=False)

    print("All internal states csv have been transposed.")
    
if __name__ == '__main__':
    group_dimensions_for_plot(sys.argv[1], sys.argv[2])
