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
                    out_path = path0 + "/" + dir2.replace("1950-2014.nc","1950-2014-timmean.nc")
                    cmd = "cdo timmean " + file_path + " " + out_path
                    print(cmd)
                    # os.system(cmd)