# -- coding:utf-8 --

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

reference_region1=gpd.read_file("C:/Users/Jin/Desktop/IPCC6_Subregions/CNA/CNA.shp")

# reference_region1.plot(cmap='jet',edgecolor='black',column='Continent')
# plt.show()
# #Importing an ESRI Shapefile and plotting it using GeoPandas
# reference_region=gpd.read_file("F:/PR_attribution3_indices/IPCC_WetDry/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
# # print(reference_region.shape)
# # print(reference_region.head())
# # where jet comes from----matplotlib library
# reference_region.plot(facecolor="none",edgecolor='black',column='Continent')
# plt.show()


continental_region = gpd.read_file("F:/PR_attribution3_indices/IPCC_WetDry/IPCC-WGII-continental-regions_shapefile/IPCC_WGII_continental_regions.shp")
fig, ax = plt.subplots(figsize=(21, 7))
continental_region.plot(ax=ax, facecolor='none', edgecolor='black', linewidth = 2)
# 设置横纵坐标名称和大小
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 40
        }
bwith = 4  # 边框宽度设置为2
TK = plt.gca()  # 获取边框
TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
TK.spines['left'].set_linewidth(bwith)  # 图框左边
TK.spines['top'].set_linewidth(bwith)  # 图框上边
TK.spines['right'].set_linewidth(bwith)  # 图框右边
plt.ylim(-60, 90)
plt.xlim(-180, 180)
plt.xticks(ticks=[-150,-100, -50, 0, 50, 100, 150],
                 labels=["150$^°$W", "100$^°$W", "50$^°$W", "0$^°$", "50$^°$E", "100$^°$E", "150$^°$E"], font=font)
plt.yticks(ticks=[-60, -30, 0, 30, 60, 90], labels=["60$^°$S", "30$^°$S", "EQ", "30$^°$N", "60$^°$N", "90$^°$N"], font=font)
    # close ticks label

# regionlist = ",,,,,,,,"
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 40,
        }
font1 = {'family': 'Times New Roman',
        'weight': 'bold',
         'color': "red",
        'size': 30,
        }

TRP = plt.Rectangle((-179, -30), 358, 59, linewidth=4, edgecolor='green', facecolor='none', linestyle="--")
ax.add_patch(TRP)
ax.text(-175, -6, "TRP", color="green", fontdict=font)

NHM = plt.Rectangle((-179, 31), 358, 28, linewidth=4, edgecolor='blue', facecolor='none', linestyle="--")
ax.add_patch(NHM)
ax.text(-175, 41, "NHM", color="blue", fontdict=font)

NHH = plt.Rectangle((-179, 61), 358, 28, linewidth=4, edgecolor='orange', facecolor='none', linestyle="--")
ax.add_patch(NHH)
ax.text(-175, 71, "NHH", color="orange", fontdict=font)

NAM = plt.Rectangle((-110, 7.5), 30, 15, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(NAM)
ax.text(-108.5, 11, "NAM", fontdict=font1)

SAM = plt.Rectangle((-70, -25), 30, 15, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(SAM)
ax.text(-67.5, -21.5, "SAM", fontdict=font1)

NAF = plt.Rectangle((-30, 5), 60, 10, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(NAF)
ax.text(-11, 6.5, "NAF", fontdict=font1)

SAF = plt.Rectangle((25, -25), 45, 17.5, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(SAF)
ax.text(36, -20, "SAF", fontdict=font1)

IN = plt.Rectangle((70, 10), 35, 20, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(IN)
ax.text(81, 16, "IN", fontdict=font1)

EA = plt.Rectangle((110, 22.5), 25, 22.5, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(EA)
ax.text(115, 30, "EA", fontdict=font1)

AUS = plt.Rectangle((110, -20), 40, 15, linewidth=3, edgecolor='red', facecolor='none', linestyle="-")
ax.add_patch(AUS)
ax.text(122, -16, "AUS", fontdict=font1)


# 设置全局字体
plt.rc('font', family='Times New Roman', size=40)
# 设置标题字体和大小

continental_region = gpd.read_file("F:/PR_attribution3_indices/IPCC_WetDry/IPCC-WGII-continental-regions_shapefile/IPCC_WGII_continental_regions.shp")
fig.savefig("G:/1_OrigionalData/4_AE_SpatialTrend/4_TimeSeries/Subregion-map.png", transparent=True,bbox_inches='tight', pad_inches=0.1)
plt.show()