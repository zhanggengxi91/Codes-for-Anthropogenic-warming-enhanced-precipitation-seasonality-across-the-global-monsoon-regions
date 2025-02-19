# -- coding:utf-8 --
import netCDF4 as nc
import os
from math import log
from scipy.stats import genextreme as gev
import sys
import time


import os

if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/3_AENew/pr-year"
    for dir1 in os.listdir(rootpath):
        path1 = rootpath + "/" + dir1
        for dir2 in os.listdir(path1):
            if dir2.endswith(".nc") and (not dir1.startswith("._")):
                file_path = path1 + "/" + dir2
                file_pathAE = path1.replace("/pr-year","/AE-year") + "/" + dir2.replace("pr_yr","AE_yr")
                print(file_pathAE)
                out_path1 = rootpath + "/" + dir2.replace("pr_yr_","prMax_yr_")
                print("+++++++++++++++++++++++++++++开始运行++++++++++++++++++++++++++")
                print("第一步 Max Pr")
                cmd_maxpr = "cdo -timmax -fldmax " + file_path + " " + out_path1
                if cmd_maxpr:
                    print(cmd_maxpr)
                    os.system(cmd_maxpr)
                else:
                    print("find something wrong")
                print("第二步 除以prmax")
                ncdata = nc.Dataset(out_path1)
                out_path_D = rootpath + "/" + dir2.replace("pr_yr_","D_yr_")
                prmax = ncdata['pr'][0, 0, 0]
                print(prmax)
                cmd_div = " cdo divc," + str(prmax) + " " + file_path + " " + out_path_D
                if cmd_div:
                    print(cmd_div)
                    os.system(cmd_div)
                else:
                    print("find something wrong")
                os.remove(out_path1)
                print("第三步 乘以AE")
                out_path_AENew = rootpath.replace("/pr-year","") + "/" + dir2.replace("pr_yr_", "AENew_yr_")
                cmd_mul = " cdo mul " + out_path_D + " " + file_pathAE + " " + out_path_AENew
                if cmd_mul:
                    print(cmd_mul)
                    os.system(cmd_mul)
                else:
                    print("find something wrong")
                os.remove(out_path_D)



