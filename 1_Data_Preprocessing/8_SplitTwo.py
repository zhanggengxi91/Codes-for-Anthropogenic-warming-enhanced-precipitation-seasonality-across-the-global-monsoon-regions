import os

if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/4_AE_SpatialTrend/3_Spatial_Draw/Data"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir1.startswith("._")):
                    file_path = path1 + "/" + dir2
                    out_path1 = path0 + "/" + dir2.replace("_1950-2014","_1950-1979")
                    out_path2 = path0 + "/" + dir2.replace("_1950-2014", "_1980-2014")
                    cmd1 = "cdo selyear,1950/1979 " + file_path + " " + out_path1
                    cmd2 = "cdo selyear,1980/2014 " + file_path + " " + out_path2
                    print(cmd1)
                    print(cmd2)
                    os.system(cmd1)
                    os.system(cmd2)