import json
import os
import sys
import matplotlib.pyplot as plt

import json
import os
import matplotlib.pyplot as plt

import json
import os
import matplotlib.pyplot as plt


def map(json_folder_path: str,
        X_axis: str,
        Y_axis: str,
        save_folder_path: str, 
        title = "Entropy map"
        ):
    """
    X_axis y Y_axis pueden tomar cualquier valor de los siguientes:
    - Length
    - Density
    - LZ76
    - Shannon Entropy Over Factor sizes
    - Shannon Divergence
    - MaxPhraseSize
    - AveragePhraseSize
    - Asymptotic entropy density
    - Fitted entropy density
    - Excess entropy
    - Multi information
    """

    filelist = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]

    # Lista predefinida de colores
    color_list = ["purple"] * len(filelist)
    
    for index, _file in enumerate(filelist):
        list1 = []
        list2 = []
        with open(f"{json_folder_path}/" + _file) as f:
            data = json.load(f)
            list1.append(eval(data[X_axis]))
            list2.append(eval(data[Y_axis]))
            
        # Usa el color de la lista de colores
        plt.scatter(list1, list2, color=color_list[index % len(color_list)]) 

    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    plt.title(title)
            
    # Guardar la trama usando el nombre del directorio
    folder_name = os.path.basename(json_folder_path)  # Obtener el nombre de la carpeta
    filename = f"{folder_name}.png"
    plt.savefig(os.path.join(save_folder_path, filename))
    plt.clf()  # Limpiar la figura actual después de guardar


if __name__ == '__main__':
    map(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

