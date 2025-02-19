# -- coding:utf-8 --

import numpy as np
import os
import netCDF4 as nc
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps

def plot(X, Y, trend, dir3):
    fig, ax = plt.subplots(figsize=(21, 7), dpi=100)
    title = dir3.split("-")[0].replace("_","-")
    print(title)
    # 设置全局字体
    plt.rc('font', family='Times New Roman', size=40)
    # 设置标题字体和大小
    ax.set_title("        " + title.replace("ssp126", "SSP1-2.6").replace("ssp245", "SSP2-4.5").replace("ssp585", "SSP5-8.5"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 45}, loc="left", pad=20)
    colorbarRange = np.linspace(0, 100, 11, endpoint=True)
    # Define the original colormap
    cmap_original = plt.cm.gist_ncar_r

    # Define the range of colors you want to extract
    start_value = 0.01  # Start position (between 0 and 1)
    end_value = 0.92  # End position (between 0 and 1)

    # Extract the subset of colors from the original colormap
    colors_subset = cmap_original(np.linspace(start_value, end_value, 256))
    cmap_subset = ListedColormap(colors_subset)

    ax.contourf(X, Y, trend, cmap=cmap_subset, levels=colorbarRange, extend='both', zorder=90)
    p1 = ax.contourf(X, Y, trend, cmap=cmap_subset, levels=colorbarRange, extend='both', zorder=90)
    # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.0f', orientation="horizontal", extend='Neither', aspect=80,fraction=0.49, pad=0.4)
    # cbar.ax.tick_params(labelsize=25)
    # cbar.ax.set_xlabel("Relative Change (%)", fontsize=25)
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
    plt.ylim(-40, 40)
    plt.xlim(-180, 180)
    plt.xticks(ticks=[-150,-100, -50, 0, 50, 100, 150],
                 labels=["150$^°$W", "100$^°$W", "50$^°$W", "0$^°$", "50$^°$E", "100$^°$E", "150$^°$E"], font=font)
    plt.yticks(ticks=[-40, -20, 0, 20, 40, 60], labels=["40$^°$S", "20$^°$S", "EQ", "20$^°$N", "40$^°$N", "60$^°$N"], font=font)
    # close ticks label
    plt.tick_params(labelsize=40)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    region_overlay = gpd.read_file(
        "F:/PR_attributionPlot1_SpatialTrend/3_Trend/3_Draw_Picture1/PictureSpatial2_LatShift//World_Continents/World_Continents.shp")
    region_overlay.plot(ax=ax, color="none", edgecolor='black', zorder=100, alpha=0.8, linewidth=2.5)
    plt.savefig("G:/2_Monsoon_Distribution-Trend/5_SSPFuture/SpatialPercent/SSPs-Spatial_" + title  + '.png',
        format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.show()


if __name__ == '__main__':
    rootpath = "G:/2_Monsoon_Distribution-Trend/5_SSPFuture/SpatialPercent/DrawSpatial-SSPs"
    for dir3 in os.listdir(rootpath):
        if dir3.endswith(".nc") and not dir3.startswith("._"):
            path3 = rootpath + "/" + dir3
            ncdata = nc.Dataset(path3)
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
                    trend[i, j] = ncdata[zhishu][:, i, j]*100
            try:
                plot(ncdata["lon"][:], ncdata["lat"][:], trend, dir3)
            except:
                plot(ncdata["longitude"][:], ncdata["latitude"][:], trend, dir3)
