import os
import csv
import json


def window(folder, threshold):
    dict = {}
    for i in os.listdir(folder):
        sub = os.path.join(folder, i, 'rmse')
        dict[i] = {}
        for j in os.listdir(sub):
            n_sub = os.path.join(sub, j)
            dict[i][j] = search_threshold_value(n_sub, threshold)
        dict[i]['all'] = sum(dict[i].values())/len(dict[i].values())
    # dict["all"] = sum(dict[i].values())/len(dict[i].values())
    dict["all"] = sum([dict[i]['all'] for i in dict.keys()])/len(dict)

    with open(os.path.join('/home/lauren/Documentos/', 'prediction_window.json'), 'w') as f:
        json.dump(dict, f, indent=4, sort_keys=True, separators=(",", ": "))


def search_threshold_value(file, threshold):
    with open(file, 'r') as f:
        data = csv.reader(f)
        for j, fil in enumerate(data):
            for elem in fil:
                if float(elem) >= threshold:
                    return j


folder = '/home/lauren/Documentos/results'
threshold = 0.1

window(folder, threshold)
