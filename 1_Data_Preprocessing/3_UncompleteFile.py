# -- coding:utf-8 --
import os
import netCDF4 as nc


def get_file_info(rootpath):
    res1 = {}
    res2 = {}
    for dir2 in os.listdir(rootpath):
        if not dir2.startswith("."):
            path2 = rootpath + "/" + dir2
            if os.path.isdir(path2):
                for dir3 in os.listdir(path2):
                    key = dir2.split("_")[-1] + "_" + dir3
                    path3 = path2 + "/" + dir3
                    if os.path.isdir(path3):
                        temp1 = []
                        temp2 = []
                        for file_name in os.listdir(path3):
                            if "_r1i1p1f1_" not in file_name.lower():
                                print(key + " 中存在 非 r1i1p1f1 文件，需删除")
                                continue
                            if not file_name.startswith(".") and file_name.endswith(".nc"):
                                temp2.append(path3 + "/" + file_name)
                                left = int(file_name.split(".")[0].split("_")[-1].split("-")[0])
                                right = int(file_name.split(".")[0].split("_")[-1].split("-")[1])
                                temp1.append(left)
                                temp1.append(right)
                        temp1.sort()
                        res1[key] = temp1
                        res2[key] = temp2
    return res1, res2


def check_zhongjian(key, years):
    if len(years) >= 2:
        for i in range(len(years) - 2):
            if i % 2 == 0 and i > 0:
                if (int(str(years[i])[0:4]) - int(str(years[i - 1])[0:4])) == 1:
                    pass
                else:
                    return False, key + " 中间文件缺失，需补充"
    return True, "阔以"


def check_shou(key, years):
    if key.lower().startswith("ssp"):
        year_min = min(years)
        if year_min < 20150101 and year_min != 20141231:
            msg = key + " 起始年份有点低哦，需裁剪"
            flag = False
        elif year_min == 20150101 or year_min == 20141231:
            msg = "阔以"
            flag = True
        else:
            msg = key + " 起始年份有点高哦，需补充"
            flag = False
        return flag, msg
    else:
        year_min = min(years)
        if year_min < 19510101 and year_min != 19501231:
            msg = key + " 起始年份有点低哦，需裁剪"
            flag = True
        elif year_min == 19510101 or year_min == 19501231:
            msg = "阔以"
            flag = True
        else:
            msg = key + " 起始年份有点高哦，需补充"
            flag = False
        return flag, msg


def check_wei(key, years):
    if key.lower().startswith("ssp"):
        year_max = max(years)
        if year_max > 21001231 and year_max != 21010101:
            msg = key + " 终止年份有点高哦，需裁剪"
            flag = True
        elif year_max == 21001231 or year_max == 21010101:
            msg = "阔以"
            flag = True
        else:
            msg = key + " 终止年份有点低哦，需补充"
            flag = False
        return flag, msg
    else:
        year_max = max(years)
        if year_max > 20141231 and year_max != 20150101:
            msg = key + " 终止年份有点高哦，需裁剪"
            flag = True
        elif year_max == 20141231 or year_max == 20150101:
            msg = "阔以"
            flag = True
        else:
            msg = key + " 终止年份有点低哦，需补充"
            flag = False
        return flag, msg


if __name__ == '__main__':
    rootpath = "G:\PR_datadownload_SSP126"
    file_info1, file_info2 = get_file_info(rootpath)
    for key in file_info1:
        if file_info1[key]:
            shouflag, shoumsg = check_shou(key, file_info1[key])
            if not shouflag:
                print(shoumsg)
            weiflag, weimsg = check_wei(key, file_info1[key])
            if not weiflag:
                print(weimsg)
            zhongjianflag, zhongjianmsg = check_zhongjian(key, file_info1[key])
            if not zhongjianflag:
                print(zhongjianmsg)
            for file in file_info2[key]:
                try:
                    nc.Dataset(file)
                except:
                    print(key + " 文件 " + file.split("/")[-1] + " 疑似下载不完整，请查验")
        else:
            print(key + " 路径下无数据")
