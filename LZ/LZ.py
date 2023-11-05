import os
import sys
import re
import json

def lzProc(root_folder_path: str, save_folder_path: str):   
    filelist = os.listdir(root_folder_path)

    base_name = os.path.basename(root_folder_path)
    lz76_path = os.path.join(f"{save_folder_path}/LZ76/", base_name)
    json_path = os.path.join(f"{save_folder_path}/json/", base_name)

    if not os.path.exists(lz76_path):
        os.makedirs(lz76_path)
    
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    for _file in filelist:
        if _file.endswith(".txt"):

            if _file + ".lz76" not in filelist:     
                os.system(f"LempelZiv -bauto {root_folder_path}/" + _file)
                print(f"Processing {_file}")

                os.system(f"mv {root_folder_path}/" + _file + ".lz76" + " " + f"{lz76_path}/" + _file.replace(".txt",".lz76"))
                print("Move lz76 file to save folder")

    data_extract(lz76_path, json_path)

def data_extract(lz76_folder_path: str, json_folder_path: str):
        
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
                Name = data["Input file"].split('/')[-1]

                with open(f"{json_folder_path}/" + Name + ".json", "w") as f:
                    json.dump(data, f)


if __name__ == "__main__":
    lzProc(sys.argv[1], sys.argv[2])

