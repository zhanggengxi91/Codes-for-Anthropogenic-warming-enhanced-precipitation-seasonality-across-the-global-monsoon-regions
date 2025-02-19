# -- coding:utf-8 --
import os
from collections import defaultdict


def get_file_info(rootpath):
    res = defaultdict(list)
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir2.startswith("._")):
                    path2 = path1 + "/" + dir2
                    res[dir1 + "#" + dir2.split("_")[0]].append(path2)
    return res


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/2_AE_RegridMask/AE"
    topopath = "G:/1_OrigionalData/2_AE_RegridMask/topo_r360x180.nc"
    file_info = get_file_info(rootpath)
    for key in file_info.keys():
        for filepath in file_info[key]:
            print("+++++++++++++++++++++++++++++开始运行++++++++++++++++++++++++++")
            print("第一步 remapbil")
            cmd_remapbil = ""
            out_path_remapbil = "/".join(filepath.split("/")[0:len(filepath.split("/")) - 1]) + "/" + "remapbil_" + \
                                filepath.split("/")[-1]
            cmd_remapbil = "cdo remapbil," + topopath + " -selname,pr " + filepath + " " + out_path_remapbil

            if cmd_remapbil:
                print(cmd_remapbil)
                # os.system(cmd_remapbil)
            else:
                print("find something wrong")
            print("第二步 删除 bnd")
            out_path_delete = "/".join(filepath.split("/")[0:len(filepath.split("/")) - 1]) + "/" + "delete_" + \
                              filepath.split("/")[-1]
            cmd_delete = "ncks -C -O -x -v time_bnds,time_bnds_2,lat_bnds,lon_bnds " + out_path_remapbil + " " + out_path_delete
            print(cmd_delete)
            # os.system(cmd_delete)
            print("删除中间文件")
            # os.remove(out_path_remapbil)
            print("第三步 相乘")
            cmd_mul = ""
            out_path_final = rootpath + "/" + filepath.split("/")[-1]
            cmd_mul = " cdo -sellonlatbox,-180,180,-60,90 -mul " + topopath + " " + out_path_delete + " " + out_path_final.replace("AE_","AE-land_")

            if cmd_mul:
                print(cmd_mul)
                # os.system(cmd_mul)
            else:
                print("find something wrong")
            print("删除中间文件")
            # os.remove(out_path_delete)




