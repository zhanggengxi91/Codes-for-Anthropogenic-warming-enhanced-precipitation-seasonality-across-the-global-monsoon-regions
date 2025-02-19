# -- coding:utf-8 --
import os


def year_split(rootpath, dir1, gap):
    filepath = rootpath + "/" + dir1
    temp_filepath = dir1[0:len(dir1) - 20]
    if dir1.endswith(".nc") and (not dir1.startswith("._")):
        time_0 = int(dir1[-20:-3].split("-")[0][0:4])
        time_1 = int(dir1[-20:-3].split("-")[1][0:4])
        temp = time_0
        while temp <= time_1:
            time_left = str(temp) + "0101"
            time_right = str(temp + (gap - 1)) + "1231"
            if temp + (gap - 1) > time_1:
                time_right = str(time_1) + "1231"
            outputpath = rootpath.replace(rootpath.split("/")[-1], "") + temp_filepath + time_left + "-" + time_right + ".nc"
            cmd = "cdo selyear," + str(temp) + "/" + str(temp + (gap - 1)) + " " + filepath + " " + outputpath
            print(cmd)
            os.system(cmd)
            temp = temp + (gap - 1) + 1


if __name__ == '__main__':
    # nc文件所在目录
    rootpath = "/mnt/g/1"
    # 年份间隔
    gap = 65
    for dir1 in os.listdir(rootpath):
        path1 = rootpath + "/" + dir1
        year_split(rootpath, dir1, gap)
