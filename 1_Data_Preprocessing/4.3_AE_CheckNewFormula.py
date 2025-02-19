# -- coding:utf-8 --
import netCDF4 as nc
import os
from math import log
from scipy.stats import genextreme as gev
import sys
import time


def check_nc_file(filepath):
    ncdata = nc.Dataset(filepath)
    zhishu = "pr"
    lat_num = len(ncdata["lat"][:])
    lon_num = len(ncdata["lon"][:])
    for k in range(lon_num):
        for j in range(lat_num):
            datas = ncdata[zhishu][:, j, k]
            for data in datas:
                if float(data) < 0.0:
                    print(float(data))
                    return filepath
    return ""

if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/2_AE_RegridMask/1_AE/AE-entropy _addlog12"
    for dir1 in os.listdir(rootpath):
        path1 = rootpath + "/" + dir1
        for dir2 in os.listdir(path1):
            if dir2.endswith(".nc") and (not dir1.startswith("._")):
                file_path = path1 + "/" + dir2
                try:
                    if check_nc_file(file_path):
                        print(file_path)
                except Exception as e:
                    print(e)
                    print(dir1 + " processing error, please check!")





#for one file path
# # -- coding:utf-8 --
# import netCDF4 as nc
# import os
# from scipy.stats import genextreme as gev
# import sys
# import time
#
#
# def check_nc_file(filepath):
#     print(filepath)
#     ncdata = nc.Dataset(filepath)
#     zhishu = filepath.split("/")[-1].split("_")[0]
#     lat_num = len(ncdata["lat"][:])
#     lon_num = len(ncdata["lon"][:])
#     for k in range(lon_num):
#         for j in range(lat_num):
#             datas = ncdata[zhishu][:, j, k]
#             for data in datas:
#                 if float(data) > 1.0 or float(data)  < 0:
#                     print(dir1 + " processing error, please check!")
#                     sys.exit()
#
#
# if __name__ == '__main__':
#     rootpath = "C:/FinishedPI/hist-aer/DJF"
#     for dir1 in os.listdir(rootpath):
#         if dir1.endswith(".nc") and (not dir1.startswith("._")):
#             path1 = rootpath + "/" + dir1
#             try:
#                 check_nc_file(path1)
#             except Exception as e:
#                 print(e)
#                 print(dir1 + " processing error, please check!")
