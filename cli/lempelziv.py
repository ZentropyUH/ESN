from typer import Typer
from typer import Option
from typing import List
from cli.enums import *
from lempelziv.lz_utils import *

app = Typer(
    help="",
    no_args_is_help=True,
)

# binary_sequences_from_csv
# lz_csv
# lz_distance_csv
# lz_folder
# distance_matrices_single_folder
# distance_matrices_between_folders


@app.command(
    name="binary-csv",
    help="Binarize a csv file by column and save the binary sequences to a folder.",
    no_args_is_help=True,
)
def binarize_csv(
    filepath: str = Option(
        ...,
        "--filepath",
        "-f",
        help="Path to the csv file."
        ),
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the binary sequences.",
        ),
    ):
    from lempelziv.lz_utils import binary_sequences_from_csv
    
    binary_sequences_from_csv(
        filepath=filepath,
        method=method,
        save_path=save_path,
        )

@app.command(
    name="lz-csv",
    help="Compute the Lempel-Ziv complexity of a csv file by column and save the results to a folder.",
    no_args_is_help=True,
)
def process_csv(
    filepath: str = Option(
        ...,
        "--filepath",
        "-f",
        help="Path to the csv file."
        ),
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the binary sequences.",
        ),
    ):

    from lempelziv.lz_utils import lz_csv
    
    lz_csv(
        file_path=filepath,
        method=method,
        save_path=save_path,
        )
    
@app.command(
    name="lz-distance-csv",
    help="Compute the Lempel-Ziv complexity of a csv file by column and save the results to a folder.",
    no_args_is_help=True,
)
def distance_csv(
    csv1: str = Option(
        ...,
        "--csv1",
        "-c1",
        help="Path to the first csv file."
        ),
    csv2: str = Option(
        ...,
        "--csv2",
        "-c2",
        help="Path to the second csv file."
        ),
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the binary sequences.",
        ),
    ):
    from lempelziv.lz_utils import lz_distance_csv
    
    print(lz_distance_csv(
        csv_file1=csv1,
        csv_file2=csv2,
        method=method,
        ))
    
    
@app.command(
    name="lz-folder",
    help="Compute the Lempel-Ziv complexity of a folder with txt files and save the results to a folder.",
    no_args_is_help=True,
)
def process_folder(
    folder_path: str = Option(
        ...,
        "--folder-path",
        "-f",
        help="Path to the folder with txt files."
        ),
    
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the results.",
        ),
    ):
    from lempelziv.lz_utils import lz_folder
    
    lz_folder(
        folder_path=folder_path,
        method=method,
        save_path=save_path,
        )
    
@app.command(
    name="distance-matrices-single-folder",
    help="Compute the Lempel-Ziv complexity of a folder with txt files and save the results to a folder.",
    no_args_is_help=True,
)
def process_single_folder(
    folder_path: str = Option(
        ...,
        "--folder-path",
        "-f",
        help="Path to the folder with txt files."
        ),
    
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the results.",
        ),
    ):
    from lempelziv.lz_utils import distance_matrices_single_folder
    
    distance_matrices_single_folder(
        folder_path=folder_path,
        method=method,
        save_path=save_path,
        )

@app.command(
    name="distance-folders",
    help="Compute the Lempel-Ziv complexity of a folder with txt files and save the results to a folder.",
    no_args_is_help=True,
)
def process_two_folders(
    folder_path1: str = Option(
        ...,
        "--folder-path1",
        "-f1",
        help="Path to the first folder with txt files."
        ),
    
    folder_path2: str = Option(
        ...,
        "--folder-path2",
        "-f2",
        help="Path to the second folder with txt files."
        ),
    
    method: EnumBinMethod = Option(
        "mean",
        "--method",
        "-m",
        help="Method to binarize the csv file.",
        show_default=True,
        case_sensitive=False,
        ),
    
    save_path: str = Option(
        ...,
        "--save-path",
        "-s",
        help="Path to save the results.",
        ),
    ):
    from lempelziv.lz_utils import distance_matrices_between_folders
    
    distance_matrices_between_folders(
        folder_path1=folder_path1,
        folder_path2=folder_path2,
        method=method,
        save_path=save_path,
        )
    