# -- coding:utf-8 --
import os
from collections import defaultdict


def get_file_info(rootpath):
    res = defaultdict(list)
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2
        if os.path.isdir(path2):
            for dir3 in os.listdir(path2):
                path3 = path2 + "/" + dir3
                res[dir2].append(path3)

    return res


def get_final_path(resutl_rootpath, tag):
    regionlist = "TRP,NHM,NHH,NAM,SAM,NAF,SAF,IN,EA,WNP,AUS"
    region = list(regionlist.split(","))
    templist = ((-180, 180, -30, 30), (-180, 180, 30, 60), (-180, 180, 60, 90), (-110, -80, 7.5, 22.5), (-70, -40, -25, -5),
    (-30, 30, 5, 15), (25, 70, -25, -7.5), (70, 105, 10, 30), (110, 135, 22.5, 45), (110, 150, 12.5, 22.5),(110, 150, -20, -5))
    temp = list(templist)
    for i in range(len(region)):
        temp_path_0 = resutl_rootpath + "/" + str(region[i])
        if not os.path.exists(temp_path_0):
            os.mkdir(temp_path_0)
        for key in tag:
            temp_path_1 = temp_path_0 + "/" + key
            if not os.path.exists(temp_path_1):
                os.mkdir(temp_path_1)


def exec_cmd(infile_path, tag, resutl_rootpath):
    regionlist = "TRP,NHM,NHH,NAM,SAM,NAF,SAF,IN,EA,WNP,AUS"
    region = list(regionlist.split(","))
    templist = ((-180, 180, -30, 30), (-180, 180, 30, 60), (-180, 180, 60, 90), (-110, -80, 7.5, 22.5), (-70, -40, -25, -5),
    (-30, 30, 5, 15), (25, 70, -25, -7.5), (70, 105, 10, 30), (110, 135, 22.5, 45), (110, 150, 12.5, 22.5),(110, 150, -20, -5))
    temp = list(templist)
    for i in range(len(temp)):
        temp_outpath = resutl_rootpath + "/" + str(region[i]) + "/" + tag \
                       + "/" + str(infile_path).split("/")[-1].replace("land_", str(region[i] + "_"))
        print(infile_path)
        temp_cmd = "cdo sellonlatbox," + str(temp[i][0]) + "," + str(temp[i][1]) + "," + str(temp[i][2]) + "," +str(temp[i][3]) + " "  +infile_path + " " + temp_outpath
        if os.path.exists(temp_outpath):
            os.remove(temp_outpath)
        print(temp_cmd)
        os.system(temp_cmd)


if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/3_SI-Regided-OceanMask/SI-Ready"
    resutl_rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/4_SI_Subregion"
    file_info = get_file_info(rootpath)
    get_final_path(resutl_rootpath, file_info.keys())
    for key in file_info.keys():
        for infile_path in file_info[key]:
            exec_cmd(infile_path, key, resutl_rootpath)
