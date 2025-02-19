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


def get_final_path(resutl_rootpath, tag, index):
    temp = list(range(-60, 91, index))
    for i in range(len(temp) - 1):
        temp_path_0 = resutl_rootpath + "/lat_" + str(abs(temp[i])) + str(abs(temp[i + 1]))
        if not os.path.exists(temp_path_0):
            os.mkdir(temp_path_0)
        for key in tag:
            temp_path_1 = temp_path_0 + "/" + key
            if not os.path.exists(temp_path_1):
                os.mkdir(temp_path_1)


def exec_cmd(infile_path, tag, index, resutl_rootpath):
    temp = list(range(-60, 91, index))
    for i in range(len(temp) - 1):
        temp_outpath = resutl_rootpath + "/lat_" + str(abs(temp[i])) + str(abs(temp[i + 1])) + "/" + tag \
                       + "/" + str(infile_path).split("/")[-1].replace("land", str(abs(temp[i])) + str(abs(temp[i + 1])))
        temp_cmd = "cdo sellonlatbox,0,360," + str(temp[i]) + "," + str(temp[i + 1]) + " "  +infile_path + " " + temp_outpath
        if os.path.exists(temp_outpath):
            os.remove(temp_outpath)
        print(temp_cmd)
        os.system(temp_cmd)


if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/3_AENew-Regided-OceanMask"
    resutl_rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/4_AENew_Subregion"
    index = 30
    file_info = get_file_info(rootpath)
    get_final_path(resutl_rootpath, file_info.keys(), index)
    for key in file_info.keys():
        for infile_path in file_info[key]:
            exec_cmd(infile_path, key, index, resutl_rootpath)
