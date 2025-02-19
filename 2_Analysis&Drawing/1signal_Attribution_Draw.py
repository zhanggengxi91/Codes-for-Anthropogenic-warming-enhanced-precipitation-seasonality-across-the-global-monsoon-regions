# -- coding:utf-8 --
import pandas as pd
# import scikits.bootstrap as bootstraps
import numpy as np
from matplotlib import pyplot as plt
# import proplot as pplot
import cmaps
import xarray as xr
from scipy.stats import linregress
from warnings import simplefilter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.pyplot import MultipleLocator

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
# ####################   read file   #################
scaling_factor_and_trends_3sig_ribes = pd.read_csv('G:/1_OrigionalData/5_Attribution/2_csv_1985-2014/trends_scaling_factors_Ribes_GPH.csv', index_col=0)
# ####################   transfer 5 year to decade   #################
scaling_factor_and_trends_3sig_ribes['trend'] = scaling_factor_and_trends_3sig_ribes['trend'] * 200
scaling_factor_and_trends_3sig_ribes['trend_min'] = scaling_factor_and_trends_3sig_ribes['trend_min'] * 200
scaling_factor_and_trends_3sig_ribes['trend_max'] = scaling_factor_and_trends_3sig_ribes['trend_max'] * 200

plt.rc('font', family='Times New Roman', size=18)
for zhishu in ['SI']:
    for jijie in ['ANN']:
        fig, ax = plt.subplots(1, 11, sharex=False, figsize=(16, 3), gridspec_kw={'wspace': 0, 'hspace': 0.25})
        # regions = ['AllMask', 'WetRegion', 'DryRegion', 'CNA', 'EAS', 'ECA', 'EEU', 'ENA', 'ESB', 'MED', 'NCA', 'NEU', 'NWN', 'RAR', 'RFE', 'SAS', 'SCA', 'TIB', 'WAF', 'WCA', 'WCE', 'WNA', 'WSB']
        regions = ['land', 'TRP', 'NHM', 'NHH', 'NAM', 'SAM', 'NAF', 'SAF', 'IN', 'EA', 'AUS']
        color_f = ['#3CB371', '#EE3B3B', '#1874CD', '#FF9912']
        # ####################   draw scaling factor   #################
        for r in range(len(regions)):
            marker = 'o'
            domain = regions[r]

            sf_df2 = scaling_factor_and_trends_3sig_ribes[scaling_factor_and_trends_3sig_ribes['zhishu'] == zhishu]
            sf_df2 = sf_df2[sf_df2['domain'] == domain]
            sf_df2 = sf_df2[sf_df2['jijie'] == jijie]
            ax[r].axhline(y=0, color='black', alpha=0.6, linewidth=1.5)
            ax[r].axhline(y=1, linestyle=':', color='black', alpha=0.6, linewidth=1.5)
            # ####################   1-ghg 2-nat 3-aer 4-historical   #################
            ax[r].plot((2, 2),
                       (sf_df2['sf_min'][sf_df2['forcing'] == 'Forcing no 4 only'], sf_df2['sf_max'][sf_df2['forcing'] == 'Forcing no 4 only']),
                       color=color_f[0], alpha=1, linewidth=2)
            ax[r].plot((3, 3),
                       (sf_df2['sf_min'][sf_df2['forcing'] == 'Forcing no 1 only'], sf_df2['sf_max'][sf_df2['forcing'] == 'Forcing no 1 only']),
                       color=color_f[1], alpha=1, linewidth=2)
            ax[r].plot((4, 4),
                       (sf_df2['sf_min'][sf_df2['forcing'] == 'Forcing no 3 only'], sf_df2['sf_max'][sf_df2['forcing'] == 'Forcing no 3 only']),
                       color=color_f[2], alpha=1, linewidth=2)
            ax[r].plot((5, 5),
                       (sf_df2['sf_min'][sf_df2['forcing'] == 'Forcing no 2 only'], sf_df2['sf_max'][sf_df2['forcing'] == 'Forcing no 2 only']),
                       color=color_f[3], alpha=1, linewidth=2)

            ax[r].scatter(x=2, y=sf_df2['sf_best'][sf_df2['forcing'] == 'Forcing no 4 only'], marker=marker, s=55, edgecolor=color_f[0], facecolor="none", linewidth=1.5)
            ax[r].scatter(x=3, y=sf_df2['sf_best'][sf_df2['forcing'] == 'Forcing no 1 only'], marker=marker, s=55, edgecolor=color_f[1], facecolor="none", linewidth=1.5)
            ax[r].scatter(x=4, y=sf_df2['sf_best'][sf_df2['forcing'] == 'Forcing no 3 only'], marker=marker, s=55, edgecolor=color_f[2], facecolor="none", linewidth=1.5)
            ax[r].scatter(x=5, y=sf_df2['sf_best'][sf_df2['forcing'] == 'Forcing no 2 only'], marker=marker, s=55, edgecolor=color_f[3], facecolor="none", linewidth=1.5)

            # ax[r].scatter(x=2, y=sf_df2['p_cons'][sf_df2['forcing'] == 'Forcing no 4 only'], marker="1", s=60, edgecolor="gray",
            #                  facecolor="gray", linewidth=1, zorder=100)
            # ax[r].scatter(x=3, y=sf_df2['p_cons'][sf_df2['forcing'] == 'Forcing no 1 only'], marker="1", s=60, edgecolor="gray",
            #                  facecolor="gray", linewidth=1, zorder=100)
            # ax[r].scatter(x=4, y=sf_df2['p_cons'][sf_df2['forcing'] == 'Forcing no 3 only'], marker="1", s=60, edgecolor="gray",
            #                  facecolor="gray", linewidth=1, zorder=100)
            # ax[r].scatter(x=5, y=sf_df2['p_cons'][sf_df2['forcing'] == 'Forcing no 2 only'], marker="1", s=60, edgecolor="gray",
            #                  facecolor="gray", linewidth=1, zorder=100)

            ax[r].set(xlim=(1, 6))
            ax[r].set_xticks([])
            ax[r].grid(False)
            ax[r].set(ylim=(-6, 8))
            y_major_locator = MultipleLocator(2)
            ax[r].yaxis.set_major_locator(y_major_locator)
            bwith = 1.5  # 边框宽度设置为2
            ax[r].spines['right'].set_linewidth(bwith)
            ax[r].spines['left'].set_linewidth(bwith)
            ax[r].spines['top'].set_linewidth(bwith)
            ax[r].spines['bottom'].set_linewidth(bwith)

            if r > 0:
                ax[r].set_yticks([])
            if r == 0:
                ax[r].spines['right'].set_linestyle("--")
                ax[r].spines['right'].set_linewidth(0.5)
                ax[r].spines['right'].set_color("black")
                ax[r].spines['right'].set_alpha(0.5)
            if 0 < r < 11:
                ax[r].spines['left'].set_visible(False)
                ax[r].spines['right'].set_linestyle("--")
                ax[r].spines['right'].set_linewidth(0.5)
                ax[r].spines['right'].set_color("black")
                ax[r].spines['right'].set_alpha(0.5)
            # if r == 2:
            #     ax[r].spines['right'].set_linestyle("-")
            #     ax[r].spines['right'].set_linewidth(1)
            #     ax[r].spines['right'].set_color("black")
            if r == 11:
                ax[r].spines['left'].set_visible(False)
            if r == 3 or r == 10:
                ax[r].spines['right'].set_linestyle("-")
                ax[r].spines['right'].set_linewidth(1.5)
                ax[r].spines['right'].set_color("black")
                ax[r].spines['right'].set_alpha(1)

            labels = ['GHG', 'AER', 'NAT', "Historical"]

            #     ############# trend ##############3

            labels = ["Historical", 'GHG', 'AER', 'NAT']
            ax[r].set_xlabel(domain.replace("land", "GLB"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 15}, )
            ax[r].set_xticks([])

        labels = ["ALL", 'GHG', 'AER', 'NAT']
        legend_elements = [
            Line2D([0], [0], marker=marker, color=color_f[0], markeredgecolor=color_f[0], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[1], markeredgecolor=color_f[1], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[2], markeredgecolor=color_f[2], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[3], markeredgecolor=color_f[3], markerfacecolor="none", markersize=8, linewidth=2),
        ]
        handles = legend_elements

        anchor_x = -0.5
        anchor_y = 0

        ax[1].legend(handles, labels, frameon=False, ncol=8, fontsize=15, fancybox=True, loc='upper right',
                     bbox_to_anchor=(7, -0.1))  # bbox_to_anchor=[1.01,0.7],
        ax[0].text(0, 1.03, jijie.replace("ANN", " (b) Scaling Factor (1985-2014)").replace("JJA", " (c) Scaling Factor - JJA").replace("DJF", " (e) Scaling Factor - DJF").replace("SON", " (d) Scaling Factor - SON").replace("MAM", " (b) Scaling Factor - MAM"),
                   fontdict={'family': 'Times New Roman', 'fontsize': 25, 'weight': 'bold'},
                   transform=ax[0].transAxes)
        # ax[0, 0].set_title(' Scaling Factor', font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22},
        #                    loc="left", pad=0.01)
        # ax[1, 0].set_title(' Trend (%/decade)', font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22},
        #                    loc="left", pad=0.01)

        plt.tight_layout()
        fig_dir = 'G:/1_OrigionalData/5_Attribution/2_csv_1985-2014/'
        fig.savefig(fig_dir + '1985-2014_5yearMean_1singal_' + zhishu + '_' + jijie + '.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()
