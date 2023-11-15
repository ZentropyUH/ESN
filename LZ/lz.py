import os
import re
import subprocess
import shutil
import json

def lzProc(save_folder_path: str, params: tuple = ("bauto",), root_folder_path: str = ""):   
    """
    This script processes files with LempelZiv based on the provided parameters.
    The behavior depends on the nature and number of arguments passed:

    1. Flags (params[0]): 
    - The first parameter is expected to be a flag that modifies the script's behavior. 
    - Flags are optional; if no flags are provided, default behavior is assumed.

    2. Main File (params[1]): 
    - The second parameter should be the path to the main file to be processed. 
    - This parameter is optional. If omitted, the script processes all files in the specified root_folder_path.

    3. Secondary File (params[2]):
    - The third parameter can be the path to a secondary file for processing. 
    - Like the main file, this is also optional and is only considered if provided.

    Important:
    - The 'root_folder_path' variable must be set either directly or through default mechanisms. 
    - It determines the directory from which files are processed. 
    - If 'root_folder_path' is not explicitly provided, the script will not function as intended.
    """
  
    json_path = ""
    
    if len(params) == 3:
            main_file = os.path.basename(params[1])
            secundary_file = os.path.basename(params[2])

            subprocess.run(["LempelZiv", f"-{params[0]}", params[1], params[2]])

            # Create folders
            base_name = main_file.replace(".txt", "") + "_" + secundary_file.replace(".txt", "")
            lz76_path = os.path.join(f"{save_folder_path}/LZ76/", base_name)
            json_path = os.path.join(f"{save_folder_path}/json/", base_name)

            if not os.path.exists(lz76_path):
                os.makedirs(lz76_path)
                    
            if not os.path.exists(json_path):
                os.makedirs(json_path)

            # Move the .lz76 file to the output directory
            file_path = os.path.join(params[2] + ".lz76")
            output_path = os.path.join(lz76_path, base_name + ".lz76")
            shutil.move(file_path, output_path)
            
            data_extract(lz76_path, json_path, base_name)

            # return the path to de json with the data
            return os.path.join(json_path, base_name + ".json")


    if len(params) == 2:
            main_file = os.path.basename(params[1])

            subprocess.run(["LempelZiv", f"-{params[0]}", params[1]])

            # Create folders
            base_name = main_file.replace(".txt", "")
            lz76_path = os.path.join(f"{save_folder_path}/LZ76/", base_name)
            json_path = os.path.join(f"{save_folder_path}/json/", base_name)

            if not os.path.exists(lz76_path):
                os.makedirs(lz76_path)
                    
            if not os.path.exists(json_path):
                os.makedirs(json_path)

            # Move the .lz76 file to the output directory
            file_path = os.path.join(params[1] + ".lz76")
            output_path = os.path.join(lz76_path, base_name + ".lz76")
            shutil.move(file_path, output_path)

            data_extract(lz76_path, json_path)

            # return the path to de json with the data
            return os.path.join(json_path, base_name + ".json")

    if len(params) == 1:
        filelist = os.listdir(root_folder_path)
    
        for _file in filelist:
            if _file.endswith(".txt"):
                    subprocess.run(["LempelZiv", f"-{params[0]}", f"{root_folder_path}/{_file}"])

                    base_name = os.path.basename(root_folder_path)

                    lz76_path = os.path.join(f"{save_folder_path}/LZ76/", base_name)
                    json_path = os.path.join(f"{save_folder_path}/json/", base_name)

                    if not os.path.exists(lz76_path):
                        os.makedirs(lz76_path)
                    
                    if not os.path.exists(json_path):
                        os.makedirs(json_path)

                    # Move the .lz76 file to the output directory
                    file_path = os.path.join(root_folder_path, _file + ".lz76")
                    output_path = os.path.join(lz76_path, _file.replace(".txt", ".lz76"))
                    shutil.move(file_path, output_path)

                    data_extract(lz76_path, json_path)

    # return the path to de json folder with the json data   
    return json_path


def data_extract(lz76_folder_path: str, json_folder_path: str, name: str = "1"):
    data = {}
    Name = ''
    filelist = os.listdir(lz76_folder_path)
    for _file in filelist:
        print(f"Processing {_file}")
        if _file.endswith(".lz76"):
            # iteration over lz files
            with open(f"{lz76_folder_path}/" + _file) as f:
                lines = f.readlines()
                # iteration over file lines
                for line in lines:
                    # Se verifica que la línea tenga el formato adecuado
                    if ': ' in line:
                        key, value = line.split(': ')
                        key = key.replace('[', '').replace(']', '').strip()
                        value = value.replace('[', '').replace(']', '').strip()
                        data[key] = value

                Name = name if name else data["Input file"].split('/')[-1]

                final_path = f"{json_folder_path}/" + Name + ".json"

                with open(final_path, "w") as f:
                    json.dump(data, f)


if __name__ == "__main__":
    lzProc(sys.argv[1], sys.argv[2])