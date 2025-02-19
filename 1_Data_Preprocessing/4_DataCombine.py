# -- coding:utf-8 --
import os


def get_nco_params(path2, path3):
    res = []
    time = []
    temp_filepath = ""
    output_path = ""
    for dir4 in os.listdir(path3):
        if len(dir4) > 20 and dir4.endswith(".nc") and (not dir4.startswith("._")):
            res.append(path3 + "/" + dir4)
            temp_filepath = dir4[0:len(dir4) - 20]
            time_0 = int(dir4[-20:-3].split("-")[0])
            time_1 = int(dir4[-20:-3].split("-")[1])
            time.append(time_0)
            time.append(time_1)
    if time:
        output_path = path2 + "/" + temp_filepath + str(min(time)) + "-" + str(max(time)) + ".nc"
    res.sort()
    return res, output_path


if __name__ == '__main__':
    rootpath = "/mnt/F:"
    for dir1 in os.listdir(rootpath):
        if dir1.startswith("PR_"):
            path1 = rootpath + "/" + dir1
            for dir2 in os.listdir(path1):
                path2 = path1 + "/" + dir2
                if os.path.isdir(path2):
                    for dir3 in os.listdir(path2):
                        path3 = path2 + "/" + dir3
                        if os.path.isdir(path3):
                            print("+++++++++++++++++++++++++++++开始运行++++++++++++++++++++++++++")
                            print("第一步 合并数据")
                            # 第一步 合并数据
                            nco_params, output_path = get_nco_params(path2, path3)
                            if output_path:
                                cmd_cat = "cdo cat " + " ".join(nco_params) + " " + output_path
                                print(cmd_cat)
                                # os.system(cmd_cat)
                                print("第二步 年份拆分")
                                # 第二步 年份拆分
                                if dir2.lower().endswith("ssp126") or dir2.lower().endswith("ssp245") or dir2.lower().endswith("ssp585"):
                                    output_path2 = output_path.replace(output_path[-20:-3], "20150101-21001231")
                                    if output_path != output_path2:
                                        cmd_split = "cdo selyear,2015/2100 " + output_path + " " + output_path2
                                        print(cmd_split)
                                        # os.system(cmd_split)
                                        # 第三步 删除中间文件
                                        # os.remove(output_path)
                                else:
                                    output_path2 = output_path.replace(output_path[-20:-3], "19500101-20141231")
                                    if output_path != output_path2:
                                        cmd_split = "cdo selyear,1950/2014 " + output_path + " " + output_path2
                                        print(cmd_split)
                                        # os.system(cmd_split)
                                        # 第三步 删除中间文件
                                        # os.remove(output_path)
                                print("第四步 删除 bnds")
                                # 第四步 删除 pr
                                tag = "_" + output_path2.split("_")[-2] + "_"
                                output_path3 = output_path2.replace(tag, "_")
                                cmd_delete_1 = "ncks -C -O -x -v time_bnds,lat_bnds,lon_bnds " + output_path2 + " " + output_path3
                                cmd_delete_2 = "ncatted -O -a bounds,,d,, " + output_path3
                                print(cmd_delete_1)
                                print(cmd_delete_2)
                                # os.system(cmd_delete_1)
                                # os.system(cmd_delete_1)
                                # 第五步 删除中间文件
                                # os.remove(output_path2)
                                print("****************************结束运行***********************************")
