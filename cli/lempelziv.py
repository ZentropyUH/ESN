from typer import Option, Typer

from cli.enums import EnumBinMethod

app = Typer(
    help="",
    no_args_is_help=True,
)


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
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):
    from lempelziv.lz_utils import binary_sequences_from_csv
    
    binary_sequences_from_csv(
        filepath=filepath,
        method=method,
        save_path=save_path,
        max_parallel_tasks=parallel_tasks
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
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):

    from lempelziv.lz_utils import lz_csv
    
    lz_csv(
        file_path=filepath,
        method=method,
        save_path=save_path,
        max_parallel_tasks=parallel_tasks
        )
    
@app.command(
    name="distance-2-csv",
    help="Compute the Lempel-Ziv complexity of a csv file by column and save the results to a folder.",
    no_args_is_help=True,
)
def distance_csv(
    csv1: str = Option(
        ...,
        "--csv1",
        "-f1",
        help="Path to the first csv file."
        ),
    csv2: str = Option(
        ...,
        "--csv2",
        "-f2",
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
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):
    from lempelziv.lz_utils import lz_distance_csv
    import json
    
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(
            lz_distance_csv(
                csv_file1=csv1,
                csv_file2=csv2,
                method=method,
                max_parallel_tasks=parallel_tasks
                ),
            f,
            indent=4,
            )
    
    
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
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):
    from lempelziv.lz_utils import lz_folder
    
    lz_folder(
        folder_path=folder_path,
        method=method,
        save_path=save_path,
        max_parallel_tasks=parallel_tasks
        )
    

@app.command(
    name="distance-folders",
    help="Compute the Lempel-Ziv distance of the csv contained in two folders and save results to a json. If no second folder is provided the distances will be between all files of the single folder.",
    no_args_is_help=True,
)
def distance_between_folders(
    folder_path1: str = Option(
        ...,
        "--folder-path1",
        "-f1",
        help="Path to the first folder with txt files."
        ),
    
    folder_path2: str = Option(
        None,
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
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):
    from lempelziv.lz_utils import distance_matrices_between_folders
    
    distance_matrices_between_folders(
        folder_path1=folder_path1,
        folder_path2=folder_path2,
        method=method,
        save_path=save_path,
        max_parallel_tasks=parallel_tasks
        )
    
@app.command(
    name="distance-single-csv",
    help="Compute the Lempel-Ziv complexity of a folder with txt files and save the results to a folder.",
    no_args_is_help=True,
)
def distance_single_csv(
    csv_file: str = Option(
        ...,
        "--csv-file",
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
        help="Path to save the results.",
        ),
    parallel_tasks: int = Option(
        16,
        "--parallel-tasks",
        '-pt',
        help="Number of tasks to calculate simultaneously. Default is 16"
        )
    ):
    from lempelziv.lz_utils import distance_columns_single_csv
    
    distance_columns_single_csv(
        csv_file=csv_file,
        method=method,
        save_path=save_path,
        max_parallel_tasks=parallel_tasks
        )
    