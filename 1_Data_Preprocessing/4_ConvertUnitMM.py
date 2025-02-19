import os

if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/1_origionalData/1-1Monthdata_Unit_mmperM"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir1.startswith("._")):
                    time_0 = dir2[-20:-3].split("-")[0][0:4]
                    file_path = path1 + "/" + dir2
                    out_path = path0 + "/" + dir2
                    cmd = "cdo -setattribute,pr@units=mm/month -mulc,86400 -muldpm " + file_path + " " + out_path

                    print(cmd)
                    # os.system(cmd)