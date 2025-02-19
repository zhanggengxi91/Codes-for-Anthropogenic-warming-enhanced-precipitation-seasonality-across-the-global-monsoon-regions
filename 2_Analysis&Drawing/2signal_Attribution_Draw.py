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
scaling_factor_and_trends_2sig = pd.read_csv('G:/1_OrigionalData/5_Attribution/2_csv/trends_scaling_factors_2signal_GPH.csv', index_col=0)
# ####################   transfer 5 year to decade   #################
scaling_factor_and_trends_2sig['trend'] = scaling_factor_and_trends_2sig['trend'] * 200
scaling_factor_and_trends_2sig['trend_min'] = scaling_factor_and_trends_2sig['trend_min'] * 200
scaling_factor_and_trends_2sig['trend_max'] = scaling_factor_and_trends_2sig['trend_max'] * 200

plt.rc('font', family='Times New Roman', size=18)

for zhishu in ['SI']:
    for jijie in ['ANN']:
        fig, ax = plt.subplots(1, 12, sharex=False, figsize=(16, 3), gridspec_kw={'wspace': 0, 'hspace': 0.25})
        regions = ['land', 'TRP', 'NHM', 'NHH', 'NAM', 'SAM', 'NAF', 'SAF', 'IN', 'EA', 'WNP', 'AUS']
        # color_f = ['#bcbddc', '#a1d99b']
        color_f = ['#EE3B3B', '#1874CD']

        # ####################   draw scaling factor   #################
        for r in range(len(regions)):
            marker = 'o'
            domain = regions[r]
            sf_df = scaling_factor_and_trends_2sig[scaling_factor_and_trends_2sig['zhishu'] == zhishu]
            sf_df = sf_df[sf_df['domain'] == domain]
            sf_df = sf_df[sf_df['jijie'] == jijie]
            ax[r].axhline(y=0, color='black', alpha=0.6, linewidth=1.5)
            ax[r].axhline(y=1, linestyle=':', color='black', alpha=0.6, linewidth=1.5)
            ax[r].plot((1, 1), (sf_df['sf_min'][sf_df['forcing'] == 'ANT'], sf_df['sf_max'][sf_df['forcing'] == 'ANT']), color=color_f[0], alpha=1, linewidth=2)
            ax[r].plot((2, 2), (sf_df['sf_min'][sf_df['forcing'] == 'NAT'], sf_df['sf_max'][sf_df['forcing'] == 'NAT']), color=color_f[1], alpha=1, linewidth=2)
            ax[r].scatter(x=1, y=sf_df['sf_best'][sf_df['forcing'] == 'ANT'], marker=marker, s=60, edgecolor=color_f[0], facecolor="none", linewidth=2)
            ax[r].scatter(x=2, y=sf_df['sf_best'][sf_df['forcing'] == 'NAT'], marker=marker, s=60, edgecolor=color_f[1], facecolor="none", linewidth=2)

            # ax[r].scatter(x=1, y=sf_df['p_cons'][sf_df['forcing'] == 'ANT'], marker="1", s=60, edgecolor="gray", facecolor="gray",
            #               linewidth=1, zorder=100)
            # ax[r].scatter(x=2, y=sf_df['p_cons'][sf_df['forcing'] == 'NAT'], marker="1", s=60, edgecolor="gray", facecolor="gray",
            #               linewidth=1, zorder=100)

            ax[r].set(xlim=(0, 3))
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
            if 0 < r < 18:
                ax[r].spines['left'].set_visible(False)
                ax[r].spines['right'].set_linestyle("--")
                ax[r].spines['right'].set_linewidth(0.5)
                ax[r].spines['right'].set_color("black")
                ax[r].spines['right'].set_alpha(0.5)
            if r == 2:
                ax[r].spines['right'].set_linestyle("-")
                ax[r].spines['right'].set_linewidth(1)
                ax[r].spines['right'].set_color("black")
            if r == 18:
                ax[r].spines['left'].set_visible(False)
            if r == 2 or r == 7 or r == 11 or r == 17:
                ax[r].spines['right'].set_linestyle("-")
                ax[r].spines['right'].set_linewidth(1.5)
                ax[r].spines['right'].set_color("black")
                ax[r].spines['right'].set_alpha(1)

            #     ############# trend ##############3

            ax[r].set_xlabel(domain.replace("AllMask", "NHL").replace("WetRegion", "HR").replace("DryRegion", "LR"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 15}, )
            ax[r].set_xticks([])

        labels = ['ANT', 'NAT']
        legend_elements = [
            Line2D([0], [0], marker=marker, color=color_f[0], label='location', markeredgecolor=color_f[0], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[1], label='location', markeredgecolor=color_f[1], markerfacecolor="none", markersize=8, linewidth=2), ]
        handles = legend_elements
        ax[1].legend(handles, labels, frameon=False, ncol=4, fontsize=15, fancybox=True, loc='upper right', bbox_to_anchor=(10.5, -0.1))  # bbox_to_anchor=[1.01,0.7],
        ax[0].text(0, 1.03, jijie.replace("ANN", " (a) Scaling Factor - ANN").replace("JJA", " (c) Scaling Factor - JJA").replace("DJF", " (e) Scaling Factor - DJF").replace("SON", " (d) Scaling Factor - SON").replace("MAM", " (b) Scaling Factor - MAM"),
                   fontdict={'family': 'Times New Roman', 'fontsize': 25, 'weight': 'bold'},
                   transform=ax[0].transAxes)

        anchor_x = -0.5
        anchor_y = 0

        plt.tight_layout()
        fig_dir = 'G:/1_OrigionalData/5_Attribution/2_csv/'
        fig.savefig(fig_dir + '1950-2014_2singal_' + zhishu + '_' + jijie + '.png', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()
