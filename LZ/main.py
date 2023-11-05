import os

from utils import group_dimensions_for_plot
from binarize import mean_per_col_for_folder
from plotters import map
from LZ import lzProc

SISTEMA = "Lorenz" 
X_AXI = "Asymptotic entropy density"
Y_AXI = "Excess entropy"

#TODO Cambiar PATH por el path de la PC local
PATH = f"/home/sheyla/tesis/DATA/ORIGINAL/GLOBAL/{SISTEMA}"

def generate_plots(data_path: str, root_path: str):
    """
    data_path -> Ruta a la data original (carpeta con varios archivos csv)
    root_path -> Ruta a la carpeta raiz donde se crearán las demás

    Generar un mapa de entropía por dimension dado una data
    """
    #region Binarize
    bin_path = os.path.join(root_path, "Bin")
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)

    mean_per_col_for_folder(data_path, bin_path)
    #endregion

    #region LZ
    lz_path = os.path.join(root_path, "LZ")
    if not os.path.exists(lz_path):
        os.makedirs(lz_path)

    filelist = os.listdir(bin_path)
    for _file in filelist:
        lzProc(bin_path + "/" +  _file ,lz_path)
    #endregion

    #region Plot
    plot_path = os.path.join(lz_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    json_path = os.path.join(lz_path, "json")
    group_dimensions_for_plot(json_path, plot_path)

    # listar todas las carpetas de dimensiones
    filelist = os.listdir(plot_path)
    for _file in filelist:
        map(plot_path + "/" + _file , X_AXI, Y_AXI , plot_path)
    #endregion

generate_plots("/home/sheyla/tesis/DATA/ORIGINAL/GLOBAL/Lorenz/Lorenz-data", PATH)