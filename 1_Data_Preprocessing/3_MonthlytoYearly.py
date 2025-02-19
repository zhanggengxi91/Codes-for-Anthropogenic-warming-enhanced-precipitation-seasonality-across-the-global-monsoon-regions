import os

if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/1_origionalData/1-1Monthdata_Unit_mmperM"
    for dir1 in os.listdir(rootpath):
        path1 = rootpath + "/" + dir1
        for dir2 in os.listdir(path1):
            if dir2.endswith(".nc") and (not dir1.startswith("._")):
                file_path = path1 + "/" + dir2
                out_path = rootpath.replace("/month","/year") + "/" + dir2.replace("_Amon_","_yr_")
                cmd = "cdo yearsum " + file_path + " " + out_path
                print(cmd)
                # os.system(cmd)