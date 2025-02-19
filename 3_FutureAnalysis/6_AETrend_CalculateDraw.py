# -- coding:utf-8 --

import xarray
import numpy as np
import os
from scipy import stats
import netCDF4 as nc
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cmaps
import datetime

def extract_time_period(dataset, start_year, end_year):
    time_var = dataset.variables['time']
    time_units = time_var.units
    if 'years' in time_units.lower():
        time_units = time_units.replace('years', 'days')
    start_date = nc.date2num(datetime.datetime(start_year, 1, 1), time_units)
    end_date = nc.date2num(datetime.datetime(end_year, 12, 31), time_units)
    time_indices = np.where((time_var[:] >= start_date) & (time_var[:] <= end_date))[0]
    return time_indices

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def plot(X, Y, trend, p_value, dir2, dir3, rootpath, start_year, end_year):
    fig, ax = plt.subplots(figsize=(21, 7), dpi=100)
    print(dir2)
    if "observed" in dir2:
        title = str.upper(dir3.split("_")[3])
    elif "historical" in dir2:
        title = "ALL"
    else:
        title = str.upper(dir2.replace("hist-", ""))
    print(title)

    # 设置全局字体
    plt.rc('font', family='Times New Roman', size=40)

    # 设置标题字体和大小
    ax.set_title("     " + title, font={'family': 'Times New Roman', 'weight': 'bold', 'size': 45}, loc="left", pad=20)
    # cmap = plt.get_cmap('BrBG')
    cmap = cmaps.ncl_default
    # trunc_cmap = truncate_colormap(cmap, 0, 1)
    # colorbarRange = np.linspace(-0.05, 0.05, 11, endpoint=True)
    colorbarRange = np.linspace(-100, 100, 11, endpoint=True)
    p1 = ax.contourf(X, Y, trend * 100, cmap=cmap, levels=colorbarRange, extend='both', zorder=90)
    p2 = ax.contourf(X, Y, p_value, [np.min(p_value), 0.05, np.max(p_value)], hatches=['.', None], zorder=100,
                     colors="none")
    # 设置colorbar
    # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.2f', orientation="horizontal", extend='Neither', aspect=40, fraction=0.49, pad=0.4)
    # cbar.set_label('%/year', size=40, weight="bold")
    # cbar.ax.tick_params(labelsize=40)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 40

    plt.tick_params(top=False, bottom=True, left=True, right=False)
    plt.tick_params(axis="x", direction='in', which="major", length=10, width=3)
    plt.tick_params(axis="y", direction='in', which="major", length=12.5, width=2)

    bwith = 4  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    TK.spines['left'].set_linewidth(bwith)  # 图框左边
    TK.spines['top'].set_linewidth(bwith)  # 图框上边
    TK.spines['right'].set_linewidth(bwith)  # 图框右边
    # 设置横纵坐标名称和大小
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 40
            }
    plt.ylim(-40, 60)
    plt.xlim(-180, 180)
    # plt.xticks(ticks=[-180, -120, -60, 0, 60, 120, 180],
    #            labels=["180$^°$W", "120$^°$W", "60$^°$W", "0$^°$", "50$^°$E", "120$^°$E", "180$^°$E"], font=font)
    plt.xticks(ticks=[-150,-100, -50, 0, 50, 100, 150],
                 labels=["150$^°$W", "100$^°$W", "50$^°$W", "0$^°$", "50$^°$E", "100$^°$E", "150$^°$E"], font=font)
    plt.yticks(ticks=[-40, -20, 0, 20, 40, 60], labels=["40$^°$S", "20$^°$S", "EQ", "20$^°$N", "40$^°$N", "60$^°$N"], font=font)
    # close ticks label
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=40)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # 设置横纵坐标名称和大小
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 50
            }
    # 设置横纵坐标名称和大小
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 50
            }

    # plt.ylabel(key.split("#")[1], font)

    region_overlay = gpd.read_file("F:/PR_attributionPlot1_SpatialTrend/3_Trend/3_Draw_Picture1/PictureSpatial2_LatShift//World_Continents/World_Continents.shp")
    region_overlay.plot(ax=ax, color="none", edgecolor='black', zorder=100, alpha=0.8, linewidth=2.5)
    plt.savefig("G:/2_Monsoon_Distribution-Trend/1_Monsoon_SpatialTrend/" + title  + "_SI-SpatialTrend_"
                + str(start_year) + str(end_year) + '.png',
                format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.show()


if __name__ == '__main__':
    rootpath = "G:/2_Monsoon_Distribution-Trend/1_Monsoon_SpatialTrend/spatial"
    start_year = 1950
    end_year = 1980
    for dir2 in os.listdir(rootpath):
        path2 = rootpath + "/" + dir2
        for dir3 in os.listdir(path2):
            if dir3.endswith(".nc") and not dir3.startswith("._"):
                path3 = path2 + "/" + dir3
                ncdata = nc.Dataset(path3)
                time_indices = extract_time_period(ncdata, start_year, end_year)

                zhishu = "SI"
                try:
                    lat_num = len(ncdata["lat"][:])
                except:
                    lat_num = len(ncdata["latitude"][:])
                try:
                    lon_num = len(ncdata["lon"][:])
                except:
                    lon_num = len(ncdata["longitude"][:])

                trend = np.zeros((lat_num, lon_num))
                p_value = np.zeros((lat_num, lon_num))
                for i in range(0, lat_num):
                    for j in range(0, lon_num):
                        data = ncdata[zhishu][time_indices, i, j]
                        # inmdices bigger than 45 years
                        if len(data.compressed()) > 0:
                            indexs = [i for i, j in enumerate(data) if str(j) != "--"]
                            new_x = [np.arange(start_year, end_year+1)[i] for i in indexs]
                            trend[i, j], intercept, r_value, p_value[i, j], std_err = stats.linregress(new_x, data.compressed())
                        else:
                            trend[i, j], intercept, r_value, p_value[i, j], std_err = stats.linregress(np.arange(start_year, end_year+1), data)
                try:
                    plot(ncdata["lon"][:], ncdata["lat"][:], trend, p_value, dir2, dir3, rootpath, start_year, end_year)
                except:
                    plot(ncdata["longitude"][:], ncdata["latitude"][:], trend, p_value, dir2, dir3, rootpath, start_year, end_year)