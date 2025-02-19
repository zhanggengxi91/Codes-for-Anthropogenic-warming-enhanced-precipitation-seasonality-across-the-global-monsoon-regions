# -- coding:utf-8 --
import os
from collections import defaultdict


def get_file_info(rootpath):
    res = defaultdict(list)
    for dir2 in os.listdir(rootpath):
        if not dir2.startswith("."):
            path2 = rootpath + "/" + dir2
            if os.path.isdir(path2):
                for dir3 in os.listdir(path2):
                    path3 = path2 + "/" + dir3

                    res[dir2].append(path3)
    return res


def get_cmd(experiment, model_path):
    model_name = model_path.split("/")[-1]
    print(model_path)
    experiment_name = experiment.split("_")[-1].lower()
    if experiment_name.startswith("ssp"):
        yr_param = "86"
    else:
        yr_param = "-65"
    cmd = "acccmip6 -o D -m " + model_name + " -v evspsbl -f mon -r atmos -e " + experiment_name + " -dir " + model_path + " -rlzn 3 -yr " + yr_param

    # cmd = "acccmip6 -o D -m " + model_name + " -v evspsblveg -f mon -r atmos -e " + experiment_name + " -dir " + model_path + " -yr " + yr_param
    return cmd

if __name__ == '__main__':
    rootpath = "F:/PR_attribution"
    file_info = get_file_info(rootpath)
    for experiment in file_info.keys():
        for model_path in file_info[experiment]:
            cmd = get_cmd(experiment, model_path)
            print(cmd)
            os.system(cmd)
