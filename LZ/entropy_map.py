import os

from LZ.utils import group_dimensions_for_plot
from LZ.binarize import mean_per_col_for_folder
from LZ.plotters import map
from LZ.lz import lzProc

SISTEMA = "Lorenz" 
X_AXI = "Asymptotic entropy density"
Y_AXI = "Excess entropy"

#TODO Cambiar PATH por el path de la PC local
PATH = f"/home/sheyla/tesis/DATA/ORIGINAL/GLOBAL/{SISTEMA}"

def generate_plots(data_path: str, root_path: str):
    """
    Generate a dimension-wise entropy map for a given dataset.

    data_path -> Path to the original data (folder containing multiple CSV files).
    root_path -> Path to the root folder where other files will be created.
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

    folderlist = os.listdir(bin_path)
    for folder in folderlist:
        lzProc(lz_path, root_folder_path= bin_path + "/" + folder)
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

if __name__ == '__main__':
    generate_plots(sys.argv[1], sys.argv[2])
