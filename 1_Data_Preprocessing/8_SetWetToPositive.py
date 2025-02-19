import os

if __name__ == '__main__':
    rootpath = "/mnt/f/PR_attribution4_AnnPIGlobal/PI_Origional_lat090/PI_090/Wet/rx5day"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        for dir1 in os.listdir(path0):
            path1 = path0 + "/" + dir1
            for dir2 in os.listdir(path1):
                if dir2.endswith(".nc") and (not dir1.startswith("._")):
                    file_path = path1 + "/" + dir2
                    out_path = path0 + "/" + dir2
                    cmd = "cdo -mulc,-1 -setmissval,nan -selname,rx5day " + file_path + " " + out_path
                    # cmd = "cdo -sellonlatbox,-180,180,-90,90 -setmissval,nan -setctomiss,-999.0 -selname,rx5day " + file_path + " " + out_path
                    print(cmd)
                    os.system(cmd)
