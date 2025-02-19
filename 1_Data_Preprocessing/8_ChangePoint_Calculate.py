# -- coding:utf-8 --
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
from collections import defaultdict


##############       Pettitt_change_point_detection         ########################

def get_file_info(rootpath):
    res = defaultdict(list)
    for dir1 in os.listdir(rootpath):
        path1 = rootpath + "/" + dir1
        for dir2 in os.listdir(path1):
            if dir2.endswith(".nc") and not dir2.startswith("._"):
                path2 = path1 + "/" + dir2
                key = "#".join([dir2.split("_")[1], dir2.split("_")[2], dir2.split("_")[0]])
                res[key].append(dir2.split("_")[3] + "_" + dir2.split("_")[4] + "#" + path2)
    return res


def Pettitt_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2 * np.sum(r[0:x]) - x * (n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U ** 2)) / (n ** 3 + n ** 2))
    if pvalue <= 0.05:
        change_point_desc = 'Yes'
    else:
        change_point_desc = 'No'
    return K, change_point_desc


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/4_AE_SpatialTrend/4_ChangePoint"
    file_info = get_file_info(rootpath)
    for zhishu in ["rx1day", "rx5day"]:
        for key in file_info.keys():
            zhishu_temp = key.split("#")[0]
            if zhishu_temp == zhishu:
                aer = ""
                GHG = ""
                nat = ""
                mean_historical = ""
                piControl = ""
                obs_historical = ""
                for info in file_info[key]:
                    tag = info.split("#")[0]
                    if tag == "modelMean_hist-aer":
                        aer = info.split("#")[1]
                    if tag == "modelMean_hist-GHG":
                        GHG = info.split("#")[1]
                    if tag == "modelMean_hist-nat":
                        nat = info.split("#")[1]
                    if tag == "modelMean_historical":
                        mean_historical = info.split("#")[1]
                    if tag == "modelMean_piControl":
                        piControl = info.split("#")[1]
                    if tag == "observed_historical":
                        obs_historical = info.split("#")[1]
                weizhi_ner, shifouxianzhu_ner = Pettitt_change_point_detection(nc.Dataset(aer)[zhishu][:, 0, 0])
                weizhi_GHG, shifouxianzhu_GHG = Pettitt_change_point_detection(nc.Dataset(GHG)[zhishu][:, 0, 0])
                weizhi_nat, shifouxianzhu_nat = Pettitt_change_point_detection(nc.Dataset(nat)[zhishu][:, 0, 0])
                weizhi_historical, shifouxianzhu_historical = Pettitt_change_point_detection(
                    nc.Dataset(mean_historical)[zhishu][:, 0, 0])
                weizhi_piControl, shifouxianzhu_piControl = Pettitt_change_point_detection(
                    nc.Dataset(piControl)[zhishu][:, 0, 0])
                weizhi_obs, shifouxianzhu_obs = Pettitt_change_point_detection(
                    nc.Dataset(obs_historical)[zhishu][:, 0, 0])
                print(key, weizhi_ner + 1950, shifouxianzhu_ner, weizhi_GHG + 1950, shifouxianzhu_GHG,
                      weizhi_nat + 1950, shifouxianzhu_nat, weizhi_historical + 1950, shifouxianzhu_historical,
                      weizhi_piControl + 1950, shifouxianzhu_piControl, weizhi_obs + 1950, shifouxianzhu_obs)
