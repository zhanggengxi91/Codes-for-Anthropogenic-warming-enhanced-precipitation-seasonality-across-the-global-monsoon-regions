import os

if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/1_origionalData/2_YearData"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir1.startswith("._")):
                    time_0 = dir2[-20:-3].split("-")[0][0:4]
                    file_path = path1 + "/" + dir2
                    out_path = path0 + "/" + dir2.replace("pr_","pr_blank_")
                    # cmd = "cdo -selyear," + time_0 + " " + file_path + " " + out_path
                    cmd = "cdo -setmissval,nan -div " + file_path + " " + file_path + " " + out_path
                    print(cmd)
                    # os.system(cmd)