import os

if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/2_AE_RegridMask/3_AENew-Regided-OceanMask"
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2

        if os.path.isdir(path2):
            for dir3 in os.listdir(path2):
                if len(dir3) > 20 and dir3.endswith(".nc") and (not dir3.startswith("._")):
                    Date = dir3.split("_")[-1]
                    Year = dir3.split("_")[-1].replace("0101",'').replace("1231",'')
                    input_path = path2 + "/" + dir3
                    output_path1 = rootpath + "/" + dir3.replace(Date, Year + "_Timmean.nc")
                    output_path2 = rootpath + "/" + dir3.replace(Date, Year + "_Anomalies.nc")
                    output_path3 = rootpath + "/" + dir3.replace(Date, Year + "_TimeSeries.nc")

                    cmd1 = "cdo timmean " + input_path + " " + output_path1
                    cmd2 = "cdo sub " + input_path + " " + output_path1 + " " + output_path2
                    cmd3 = "cdo fldmean " + output_path2 + " " + output_path3

                    print("cmd1:   " + cmd1)
                    print("cmd1:   " + cmd2)
                    print("cmd1:   " + cmd3)


                    print("+++++++++++++++++++++++++++++")
                    os.system(cmd1)
                    os.system(cmd2)
                    os.system(cmd3)
