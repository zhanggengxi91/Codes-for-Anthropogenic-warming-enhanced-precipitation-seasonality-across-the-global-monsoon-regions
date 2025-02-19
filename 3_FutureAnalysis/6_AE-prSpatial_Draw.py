# -- coding:utf-8 --

import numpy as np
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as colors
import cmaps


# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)),
#     )
#     return new_cmap




def plot(X, Y, trend, dir3):
    fig, ax = plt.subplots(figsize=(21, 7), dpi=100)
    title = dir3.split("-")[0].replace("_","-")
    print(title)
    # 设置全局字体
    plt.rc('font', family='Times New Roman', size=40)
    # 设置标题字体和大小
    ax.set_title("     " + title, font={'family': 'Times New Roman', 'weight': 'bold', 'size': 45}, loc="left", pad=20)
    if title.split("-")[0] =="pr":
        colorbarRange = np.linspace(0, 3000, 11, endpoint=True)
        ax.contourf(X, Y, trend, cmap="terrain", levels=colorbarRange, extend='both', zorder=90)
        # p1 = ax.contourf(X, Y, trend, cmap="terrain", levels=colorbarRange, extend='both', zorder=90)
        # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.0f', orientation="horizontal", extend='Neither', aspect=40, fraction=0.49, pad=0.4)

    elif title.split("-")[0] =="AE":
        colorbarRange = np.linspace(0, 1.5, 6, endpoint=True)
        ax.contourf(X, Y, trend, cmap="gnuplot2", levels=colorbarRange, extend='both', zorder=90)
        # p1 = ax.contourf(X, Y, trend, cmap="gnuplot2", levels=colorbarRange, extend='both', zorder=90)
        # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.1f', orientation="horizontal", extend='Neither', aspect=40, fraction=0.49, pad=0.4)

    elif title.split("-")[0] == "SI":
        colorbarRange = np.linspace(0, 0.25, 6, endpoint=True)
        ax.contourf(X, Y, trend, cmap="jet", levels=colorbarRange, extend='both', zorder=90)
        # p1 = ax.contourf(X, Y, trend, cmap="jet", levels=colorbarRange, extend='both', zorder=90)
        # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.2f', orientation="horizontal", extend='Neither', aspect=40, fraction=0.49, pad=0.4)

    # ax.contourf(X, Y, trend, cmap=cmap, levels=colorbarRange, extend='both', zorder=90)
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
    # plt.xticks(ticks=[-180, -120, -60, 0, 60, 120, 180],
    #            labels=["180$^°$W", "120$^°$W", "60$^°$W", "0$^°$", "50$^°$E", "120$^°$E", "180$^°$E"], font=font)
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

    # cbar = plt.colorbar(p1, ticks=colorbarRange, format='%.0f', orientation="horizontal", extend='Neither', aspect=40, fraction=0.49, pad=0.4)
    # cbar.ax.tick_params(labelsize=40)
    plt.savefig("G:/1_OrigionalData/4_AE_SpatialTrend/3_Spatial_Draw/monsoon_Spatial/SpatialDistribution_" + title  + '.png',
        format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.show()


if __name__ == '__main__':
    rootpath = "G:/1_OrigionalData/4_AE_SpatialTrend/3_Spatial_Draw/monsoon_Spatial/2_MonsoonData"
    for dir3 in os.listdir(rootpath):
        if dir3.endswith(".nc") and not dir3.startswith("._"):
            path3 = rootpath + "/" + dir3
            ncdata = nc.Dataset(path3)
            zhishu = "pr"
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
                    trend[i, j] = ncdata[zhishu][:, i, j]
            try:
                plot(ncdata["lon"][:], ncdata["lat"][:], trend, dir3)
            except:
                plot(ncdata["longitude"][:], ncdata["latitude"][:], trend, dir3)
