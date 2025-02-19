import os
from math import log

if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/2_AE_RegridMask/1_AE"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir1.startswith("._")):
                    addlog = log(12, 2)
                    print(addlog)
                    time_0 = dir2[-20:-3].split("-")[0][0:4]
                    file_path = path1 + "/" + dir2
                    out_path = path0 + "/" + dir2
                    # cmd = "cdo -selyear," + time_0 + " " + file_path + " " + out_path
                    cmd = "cdo -mul,log2 -mulc,-1 " + file_path + " " + out_path
                    print(cmd)
                    # os.system(cmd)