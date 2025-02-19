# -- coding:utf-8 --
import os
import netCDF4 as nc
from collections import defaultdict
import pandas as pd
import numpy as np


def get_file_info(rootpath):
    res_model_ensemble = defaultdict(list)
    res_model_other = {}
    res_other = {}
    res_model_ensemble_pic = defaultdict(list)
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2
        for dir3 in os.listdir(path2):
            path3 = path2 + "/" + dir3
            if dir3 == "model":
                for dir4 in os.listdir(path3):
                    path4 = path3 + "/" + dir4
                    for dir5 in os.listdir(path4):
                        path5 = path4 + "/" + dir5
                        for dir6 in os.listdir(path5):
                            if dir6.endswith(".nc") and not dir6.startswith("._"):
                                if dir4 != "piControl":
                                    if dir6.lower().startswith("rx1day") or dir6.lower().startswith("rx5day"):
                                        key = "@".join([dir2, dir3, dir4, dir5, "AllMask", dir6.split("_")[0], dir6.split("_")[2]])
                                    else:
                                        key = "@".join([dir2, dir3, dir4, dir5, dir6.split("_")[0], dir6.split("_")[1], dir6.split("_")[3]])
                                    if dir5 == "ensemble":
                                        if dir6.lower().startswith("rx1day") or dir6.lower().startswith("rx5day"):
                                            tag = dir6.split("_")[4]
                                        else:
                                            tag = dir6.split("_")[5]
                                        res_model_ensemble[key].append(tag + "#" + path5 + "/" + dir6)
                                    else:
                                        res_model_other[key] = path5 + "/" + dir6
                                else:
                                    if dir6.lower().startswith("rx1day") or dir6.lower().startswith("rx5day"):
                                        key = "@".join([dir2, dir3, dir4, dir5, "AllMask", dir6.split("_")[0]])
                                    else:
                                        key = "@".join([dir2, dir3, dir4, dir5, dir6.split("_")[0], dir6.split("_")[1]])
                                    if dir5 == "ensemble":
                                        if dir6.lower().startswith("rx1day") or dir6.lower().startswith("rx5day"):
                                            tag = dir6.split("_")[2] + "_" + dir6.split("_")[4]
                                        else:
                                            tag = dir6.split("_")[3] + "_" + dir6.split("_")[5]
                                        res_model_ensemble_pic[key].append(tag + "#" + path5 + "/" + dir6)
            else:
                for dir4 in os.listdir(path3):
                    if dir4.endswith(".nc") and not dir4.startswith("._"):
                        path4 = path3 + "/" + dir4
                        key = "@".join([dir2, dir3, dir4.split("_")[0], dir4.split("_")[1]])
                        res_other[key] = path4
    return res_model_ensemble, res_model_other, res_other, res_model_ensemble_pic


def get_data_model_ensemble(res_model_ensemble):
    for key in res_model_ensemble.keys():
        temp = {}
        zhishu = key.split("@")[5]
        out_path = ("/".join(res_model_ensemble[key][0].split("#")[1].split("/")[0:len(res_model_ensemble[key][0].split("/")) - 1]) + "/" + key + ".csv").replace("/1_ncfile_1985-2014/", "/2_csv_1985-2014/")
        for path in res_model_ensemble[key]:
            tag = path.split("#")[0]
            title = key.split("@")[-1] + "_" + tag
            file_path = path.split("#")[1]
            ncdata = nc.Dataset(file_path)
            data = list(ncdata[zhishu][:, 0, 0])
            temp[title] = data
        df = pd.DataFrame(temp)
        df.index = np.arange(1, len(df) + 1)
        lujing = out_path.replace(out_path.split("/")[-1], "")
        if not os.path.exists(lujing):
            os.makedirs(lujing)
        if os.path.exists(out_path):
            os.remove(out_path)
        df.to_csv(out_path, index=True)


def get_data_other(res_model_other):
    for key in res_model_other.keys():
        temp = {}
        try:
            zhishu = key.split("@")[5]
        except:
            zhishu = key.split("@")[3]
        out_path = ("/".join(res_model_other[key].split("/")[0:len(res_model_other[key].split("/")) - 1]) + "/" + key + ".csv").replace("/1_ncfile_1985-2014/", "/2_csv_1985-2014/")
        title = "0"
        ncdata = nc.Dataset(res_model_other[key])
        data = list(ncdata[zhishu][:, 0, 0])
        temp[title] = data
        df = pd.DataFrame(temp)
        df.index = np.arange(1, len(df) + 1)
        lujing = out_path.replace(out_path.split("/")[-1], "")
        if not os.path.exists(lujing):
            os.makedirs(lujing)
        if os.path.exists(out_path):
            os.remove(out_path)
        df.to_csv(out_path, index=True)


def get_data_model_ensemble_pic(res_model_ensemble_pic):
    for key in res_model_ensemble_pic.keys():
        temp = {}
        zhishu = key.split("@")[5]
        out_path = ("/".join(res_model_ensemble_pic[key][0].split("#")[1].split("/")[0:len(res_model_ensemble_pic[key][0].split("/")) - 1]) + "/" + key + ".csv").replace("/1_ncfile_1985-2014/", "/2_csv_1985-2014/").replace("/ensemble/", "/")
        for i in range(len(res_model_ensemble_pic[key])):
            path = res_model_ensemble_pic[key][i]
            tag = path.split("#")[0]
            title = tag
            file_path = path.split("#")[1]
            ncdata = nc.Dataset(file_path)
            data = list(ncdata[str(zhishu)][:, 0, 0])
            temp[title] = data
        df = pd.DataFrame(temp)
        df.index = np.arange(1, len(df) + 1)
        lujing = out_path.replace(out_path.split("/")[-1], "")
        if not os.path.exists(lujing):
            os.makedirs(lujing)
        if os.path.exists(out_path):
            os.remove(out_path)
        df.to_csv(out_path, index=True)


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/5_Attribution/1_ncfile_1985-2014"
    res_model_ensemble, res_model_other, res_other, res_model_ensemble_pic = get_file_info(rootpath)
    for key in res_model_ensemble_pic.keys():
        temp_dict = defaultdict(list)
        temp_list = res_model_ensemble_pic[key]
        for path in temp_list:
            tag = path.split("#")[0]
            filepath = path.split("#")[1]
            temp_dict[tag].append(filepath)
        new_list = []
        for key1 in temp_dict.keys():
            temp_list_1 = temp_dict[key1]
            temp_list_1.sort()
            for i in range(len(temp_list_1)):
                new_list.append(key1 + "_" + str(i) + "#" + temp_list_1[i])
        res_model_ensemble_pic[key] = new_list
    get_data_model_ensemble(res_model_ensemble)
    get_data_other(res_model_other)
    get_data_other(res_other)
    get_data_model_ensemble_pic(res_model_ensemble_pic)
