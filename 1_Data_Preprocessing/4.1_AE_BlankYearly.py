import os

if __name__ == '__main__':
    rootpath = "/mnt/g/1_OrigionalData/1_origionalData/3_BlankYear/pr_full"
    for dir0 in os.listdir(rootpath):
        path0 = rootpath + "/" + dir0
        path1 = rootpath.replace("pr_full","pr_blank") + "/" + dir0.replace("AE_","pr_")
        print(path1)
        if dir0.endswith(".nc") and (not dir0.startswith("._")):
            time_0 = dir0[-20:-3].split("-")[0][0:4]
            file_path0 = path0
            file_path1 = path1
            out_path = rootpath + "/" + dir0.replace("AE_","AE_blank_")
            cmd = "cdo -div -setmissval,nan " + file_path0 + " " + file_path1 + " " + out_path
            print(cmd)
            os.system(cmd)