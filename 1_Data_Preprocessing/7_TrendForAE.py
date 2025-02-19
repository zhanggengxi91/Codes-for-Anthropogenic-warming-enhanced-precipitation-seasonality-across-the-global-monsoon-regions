import os


def getqianzhui(rootpath):
    qianzhui = []
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2
        if os.path.isdir(path2):
            for dir3 in os.listdir(path2):
                if dir3.endswith(".nc") and (not dir3.startswith("._")):
                    qianzhui.append(path2 + "/" + "_".join(dir3.split("_")[0:5]))
    return list(set(qianzhui))


def get_mean(qianzhuis):
    for qianzhui in qianzhuis:
        print(qianzhui)
        out_put_path = "/".join(qianzhui.split("/")[0:len(qianzhui.split("/")) - 2]) + "/"
        cmd = "cdo trend " + qianzhui + "_1950-2014.nc " + out_put_path + qianzhui.split("/")[4]+"_trendA.nc " + out_put_path + qianzhui.split("/")[4]+"_trendB.nc "
        print(cmd)
        # os.system(cmd)


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData¡¢3_AE_SpatialTrend"
    qianzhuis = getqianzhui(rootpath)
    get_mean(qianzhuis)
