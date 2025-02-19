# -- coding:utf-8 --
import os
import netCDF4 as nc
from scipy import stats
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

NAME_DICT1 = {"rx1day": "Rx1day", "rx5day": "Rx5day"}


def get_file_info1(rootpath1):
    res1 = defaultdict(list)
    for dir2 in os.listdir(rootpath1):
        path2 = rootpath1 + "/" + dir2
        for dir3 in os.listdir(path2):
            if not dir3.endswith("_piControl"):
                path3 = path2 + "/" + dir3
                for dir4 in os.listdir(path3):
                    if dir4.endswith(".nc") and not dir4.startswith("._"):
                        key = dir2 + "#" + dir4.split("_")[1] + "#" + dir4.split("_")[2]
                        if dir3.endswith("_HadEX3"):
                            res1[key].append(dir4.split("_")[3] + "#" + path3 + "/" + dir4)
                        else:
                            res1[key].append(dir4.split("_")[4] + "#" + path3 + "/" + dir4)
    return res1


def get_data1(zhishu1, file_paths1):
    res1 = {"s": [], "z": [], "x": []}
    if len(file_paths1) == 1:
        ncdata1 = nc.Dataset(file_paths1[0])
        res1["s"] = list(ncdata1[zhishu1][:, 0, 0] * 100)
        res1["z"] = list(ncdata1[zhishu1][:, 0, 0] * 100)
        res1["x"] = list(ncdata1[zhishu1][:, 0, 0] * 100)
    else:
        z_path = ""
        o_path = []
        for path in file_paths1:
            if "_modelMean_" in path:
                z_path = path
            else:
                o_path.append(path)
        ncdata1 = nc.Dataset(z_path)
        ncdata2 = nc.Dataset(o_path[0])
        ncdata3 = nc.Dataset(o_path[1])
        ncdata4 = nc.Dataset(o_path[2])
        ncdata5 = nc.Dataset(o_path[3])
        ncdata6 = nc.Dataset(o_path[4])
        res1["z"] = list(ncdata1[zhishu1][:, 0, 0] * 100)
        for i in range(13):
            res1["s"].append(max([float(ncdata2[zhishu1][i, 0, 0] * 100), float(ncdata3[zhishu1][i, 0, 0] * 100), float(ncdata4[zhishu1][i, 0, 0] * 100),
                                  float(ncdata5[zhishu1][i, 0, 0] * 100), float(ncdata6[zhishu1][i, 0, 0] * 100)]))
            res1["x"].append(min([float(ncdata2[zhishu1][i, 0, 0] * 100), float(ncdata3[zhishu1][i, 0, 0] * 100), float(ncdata4[zhishu1][i, 0, 0] * 100),
                                  float(ncdata5[zhishu1][i, 0, 0] * 100), float(ncdata6[zhishu1][i, 0, 0] * 100)]))
    return res1


def get_drwa_data1(key, infos1):
    zhishu = key.split("#")[1]
    res1 = defaultdict(list)
    for info in infos1:
        tag = info.split("#")[0]
        res1[tag].append(info.split("#")[1])
    obs1 = get_data1(zhishu, res1["observed"])
    hist_aer1 = get_data1(zhishu, res1["hist_aer".replace("_", "-")])
    hist_GHG1 = get_data1(zhishu, res1["hist_GHG".replace("_", "-")])
    hist_nat1 = get_data1(zhishu, res1["hist_nat".replace("_", "-")])
    historical1 = get_data1(zhishu, res1["historical".replace("_", "-")])
    return obs1, hist_aer1, hist_GHG1, hist_nat1, historical1


def get_data(path):
    ncdata = nc.Dataset(path)
    zhishu = path.split("/")[-1].split("_")[1]
    time_num = len(ncdata["time"][:])
    print(time_num)
    x = np.arange(0, time_num)
    y = ncdata[zhishu][:, 0, 0] * 100
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    def myfunc(x):
        return slope * x + intercept

    y_pred = list(map(myfunc, x))
    zhongjian = y_pred[-1] - y_pred[0]
    std = np.std(y_pred)
    confidence_interval = std * 1.96
    print(zhongjian)
    return [zhongjian - confidence_interval, zhongjian, zhongjian + confidence_interval]


def get_file_info(rootpath):
    res = defaultdict(list)
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2
        for dir3 in os.listdir(path2):
            tag = dir3.split("_")[1]
            path3 = path2 + "/" + dir3
            for dir4 in os.listdir(path3):
                if dir4.endswith(".nc") and not dir4.startswith("._"):
                    path4 = path3 + "/" + dir4
                    key = "#".join([dir2, dir4.split("_")[1], dir4.split("_")[2]])
                    res[key].append(tag + "#" + path4)
    return res


def get_draw_data(infos):
    observed = []
    hist_aer = []
    hist_GHG = []
    hist_nat = []
    historical = []
    for info in infos:
        tag = info.split("#")[0]
        path = info.split("#")[1]
        if tag == "observed":
            observed = get_data(path)
        if tag == "hist-aer":
            hist_aer = get_data(path)
        if tag == "hist-GHG":
            hist_GHG = get_data(path)
        if tag == "hist-nat":
            hist_nat = get_data(path)
        if tag == "historical":
            historical = get_data(path)
    return observed, hist_aer, hist_GHG, hist_nat, historical


def plot(key, observed, hist_aer, hist_GHG, hist_nat, historical, obs1, hist_aer1, hist_GHG1, hist_nat1, historical1, xzhou1):
    # # 设置全局字体
    plt.rc('font', family='Times New Roman', size=25)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={'width_ratios': [5, 2], 'wspace': 0})
    # bwith = 2  # 边框宽度设置为2
    # TK = plt.gca()  # 获取边框
    # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    # TK.spines['left'].set_linewidth(bwith)  # 图框左边
    # TK.spines['top'].set_linewidth(bwith)  # 图框上边
    # TK.spines['right'].set_linewidth(bwith)  # 图框右边
    # plt.tick_params(top=False, bottom=True, left=True, right=False)
    # plt.tick_params(which='both', direction='out')
    # plt.tick_params(which="major", length=7, width=1)
    # plt.tick_params(which="minor", length=4, width=1)
    # # 设置全局字体
    # plt.rc('font', family='Times New Roman', size=30)
    # plt.tight_layout()

    # plt.rc('font', family='Times New Roman', size=30)
    # plt.tight_layout()
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    # plt.tick_params(top=False, bottom=True, left=True, right=False)
    # plt.tick_params(which='both', direction='out')
    # ax1.minorticks_on()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 30
    # plt.tick_params(labelsize=30)

    bwith = 1.5  # 边框宽度设置为2
    ax1.spines['bottom'].set_linewidth(bwith)  # 图框下边
    ax1.spines['left'].set_linewidth(bwith)  # 图框左边
    ax1.spines['top'].set_linewidth(bwith)  # 图框上边
    ax1.spines['right'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(bwith)  # 图框下边
    ax2.spines['right'].set_linewidth(bwith)  # 图框左边
    ax2.spines['top'].set_linewidth(bwith)  # 图框上边
    ax2.spines['left'].set_visible(False)  # 图框右边
    ax1.tick_params(which="major", direction='out', length=7, width=1)
    ax2.tick_params(which="major", direction='out', length=7, width=1)
    ax2.tick_params(left=False)

    # ax1.minorticks_on()
    ax1.set(ylim=(-1, 1))
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set(xlim=(1950, 2015))
    ax1.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010])
    ax1.set_xticklabels([1950, 1960, 1970, 1980, 1990, 2000, 2010], rotation=45)
    # ax1.axes.xaxis.set_ticklabels([])
    # ax1.axes.yaxis.set_ticklabels([])
    ax1.grid(color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax1.yaxis.set_ticks_position('left')
    if key.split("#")[0] == "AllMask":
        ax1.set_title(" " + "(a) NHL", font={'family': 'Times New Roman', 'weight': 'bold', 'size': 30}, loc="left")
    elif key.split("#")[0] == "Dry":
        ax1.set_title(" " + "(b) LR", font={'family': 'Times New Roman', 'weight': 'bold', 'size': 30}, loc="left")
    elif key.split("#")[0] == "Wet":
        ax1.set_title(" " + "(c) HR", font={'family': 'Times New Roman', 'weight': 'bold', 'size': 30}, loc="left")
    else:
        ax1.set_title(" " + key.split("#")[0].replace("WNA", "(d) WNA").replace("CNA", "(e) CNA").replace("ENA", "(f) ENA").replace("NCA", "(g) NCA").replace("SCA", "(h) SCA")
                      .replace("NEU", "(i) NEU").replace("WCE", "(j) WCE").replace("MED", "(k) MED").replace("WAF", "(l) WAF")
                      .replace("EEU", "(m) EEU").replace("WSB", "(n) WSB").replace("WCA", "(o) WCA").replace("SAS", "(p) SAS")
                      .replace("ESB", "(q) ESB").replace("RFE", "(r) RFE").replace("EAS", "(s) EAS").replace("land", "GLB")
                      , font={'family': 'Times New Roman', 'weight': 'bold', 'size': 30}, loc="left")
    ax1.plot(xzhou1, obs1["z"], color="black", label='HadEX3', zorder=100, linewidth=2.0)
    ax1.plot(xzhou1, obs1["s"], alpha=0)
    ax1.plot(xzhou1, obs1["x"], alpha=0)
    ax1.fill_between(xzhou1, obs1["s"], obs1["x"], facecolor='black', alpha=0.1, zorder=50)

    ax1.plot(xzhou1, historical1["z"], color="red", label='ALL', zorder=100, linewidth=2.0)
    ax1.plot(xzhou1, historical1["s"], alpha=0)
    ax1.plot(xzhou1, historical1["x"], alpha=0)
    ax1.fill_between(xzhou1, historical1["x"], historical1["s"], edgecolor='red', facecolor='red', alpha=0.1, zorder=50)

    ax1.plot(xzhou1, hist_GHG1["z"], color="blue", label='GHG', zorder=100, linewidth=2.0)
    ax1.plot(xzhou1, hist_GHG1["s"], alpha=0)
    ax1.plot(xzhou1, hist_GHG1["x"], alpha=0)
    ax1.fill_between(xzhou1, hist_GHG1["s"], hist_GHG1["x"], edgecolor='blue', facecolor='blue', alpha=0.1, zorder=50)

    ax1.plot(xzhou1, hist_aer1["z"], color="darkorange", label='AER', zorder=100, linewidth=2.0)
    ax1.plot(xzhou1, hist_aer1["s"], alpha=0)
    ax1.plot(xzhou1, hist_aer1["x"], alpha=0)
    ax1.fill_between(xzhou1, hist_aer1["s"], hist_aer1["x"], edgecolor='darkorange', facecolor='darkorange', alpha=0.1, zorder=50)

    ax1.plot(xzhou1, hist_nat1["z"], color="limegreen", label='NAT', zorder=100, linewidth=2.0)
    ax1.plot(xzhou1, hist_nat1["s"], alpha=0)
    ax1.plot(xzhou1, hist_nat1["x"], alpha=0)
    ax1.fill_between(xzhou1, hist_nat1["s"], hist_nat1["x"], edgecolor='limegreen', facecolor='limegreen', alpha=0.1, zorder=50)

    # ax2.axhline(y=0, color='black', alpha=0.6, linewidth=0.8)
    # ax2.set_title(key)
    ax2.set(ylim=(-1, 1))
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.grid(color='grey', linestyle='--', linewidth=1, alpha=0.3)
    ax2.scatter([1, 1, 1], observed, marker="_", color="grey", zorder=2, linewidth=1)
    ax2.vlines([1], observed[0], observed[2], color="grey", zorder=2, linewidth=1)
    ax2.bar([1], [observed[1]], zorder=1, color="black")
    ax2.scatter([2, 2, 2], historical, marker="_", color="grey", zorder=2, linewidth=1)
    ax2.vlines([2], historical[0], historical[2], color="grey", zorder=2, linewidth=1)
    ax2.bar([2], [historical[1]], zorder=1, color="red")
    ax2.scatter([3, 3, 3], hist_GHG, marker="_", color="grey", zorder=2, linewidth=1)
    ax2.vlines([3], hist_GHG[0], hist_GHG[2], color="grey", zorder=2, linewidth=1)
    ax2.bar([3], [hist_GHG[1]], zorder=1, color="blue")
    ax2.scatter([4, 4, 4], hist_aer, marker="_", color="grey", zorder=2, linewidth=1)
    ax2.vlines([4], hist_aer[0], hist_aer[2], color="grey", zorder=2, linewidth=1)
    ax2.bar([4], [hist_aer[1]], zorder=1, color="darkorange")
    ax2.scatter([5, 5, 5], hist_nat, marker="_", color="grey", zorder=2, linewidth=1)
    ax2.vlines([5], hist_nat[0], hist_nat[2], color="grey", zorder=2, linewidth=1)
    ax2.bar([5], [hist_nat[1]], zorder=1, color="limegreen")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels(["OBS", "ALL", "GHG", "AER", "NAT"], rotation=45)
    plt.tight_layout()
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=2, prop={'size': 20, "family": "Times New Roman"})
    plt.savefig("G:/1_OrigionalData/4_AE_SpatialTrend/4_TimeSeries/2_5yearmean/" + key.replace("#", "_") + '.png',
                format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/4_AE_SpatialTrend/4_TimeSeries/2_5yearmean/Draw_Trend-Data"
    rootpath1 = "G:/1_OrigionalData/4_AE_SpatialTrend/4_TimeSeries/2_5yearmean/Draw_5yearMean-Data"
    file_info = get_file_info(rootpath)
    file_info1 = get_file_info1(rootpath1)
    for key in file_info1.keys():
        xzhou1 = list(np.linspace(1952.5, 2011.5, 13))
        obs1, hist_aer1, hist_GHG1, hist_nat1, historical1 = get_drwa_data1(key, file_info1[key])
        observed, hist_aer, hist_GHG, hist_nat, historical = get_draw_data(file_info[key])
        plot(key, observed, hist_aer, hist_GHG, hist_nat, historical, obs1, hist_aer1, hist_GHG1, hist_nat1, historical1, xzhou1)
