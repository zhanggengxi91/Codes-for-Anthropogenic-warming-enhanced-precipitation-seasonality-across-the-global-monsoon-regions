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
scaling_factor_and_trends_3sig = pd.read_csv('C:/5_attribution/res/trends_scaling_factors_3signal_GPH.csv', index_col=0)

# ####################   transfer 5 year to decade   #################
scaling_factor_and_trends_3sig['trend'] = scaling_factor_and_trends_3sig['trend'] * 200
scaling_factor_and_trends_3sig['trend_min'] = scaling_factor_and_trends_3sig['trend_min'] * 200
scaling_factor_and_trends_3sig['trend_max'] = scaling_factor_and_trends_3sig['trend_max'] * 200

plt.rc('font', family='Times New Roman', size=18)
for zhishu in ['rx1day', 'rx5day']:
    for jijie in ['ANN', 'MAM', 'JJA', 'SON', 'DJF']:
        fig, ax = plt.subplots(2, 19, sharex=False, figsize=(16, 4), gridspec_kw={'height_ratios': [2, 1], 'wspace': 0, 'hspace': 0.25})
        regions = ['AllMask', 'DryRegion', 'WetRegion', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'NEU', 'WCE', 'MED', 'EEU','WSB', 'ESB', 'RFE', 'WCA', 'EAS', 'SAS', 'WAF']
        color_f = ['#9ecae1', '#fa9fb5', '#a1d99b']
        # ####################   draw scaling factor   #################
        for r in range(len(regions)):
            marker = 'o'
            domain = regions[r]

            sf_df2 = scaling_factor_and_trends_3sig[scaling_factor_and_trends_3sig['zhishu'] == zhishu]
            sf_df2 = sf_df2[sf_df2['domain'] == domain]
            sf_df2 = sf_df2[sf_df2['jijie'] == jijie]
            ax[0, r].axhline(y=0, color='black', alpha=0.6, linewidth=1.5)
            ax[0, r].axhline(y=1, linestyle=':', color='black', alpha=0.6, linewidth=1.5)

            ax[0, r].plot((2, 2),
                          (sf_df2['sf_min'][sf_df2['forcing'] == 'GHG'], sf_df2['sf_max'][sf_df2['forcing'] == 'GHG']),
                          color=color_f[0], alpha=1, linewidth=2)
            ax[0, r].plot((3, 3),
                          (sf_df2['sf_min'][sf_df2['forcing'] == 'AER'], sf_df2['sf_max'][sf_df2['forcing'] == 'AER']),
                          color=color_f[1], alpha=1, linewidth=2)
            ax[0, r].plot((4, 4),
                          (sf_df2['sf_min'][sf_df2['forcing'] == 'NAT'], sf_df2['sf_max'][sf_df2['forcing'] == 'NAT']),
                          color=color_f[2], alpha=1, linewidth=2)


            ax[0, r].scatter(x=2, y=sf_df2['sf_best'][sf_df2['forcing'] == 'GHG'], marker=marker, s=60, edgecolor=color_f[0], facecolor="none", linewidth=2)
            ax[0, r].scatter(x=3, y=sf_df2['sf_best'][sf_df2['forcing'] == 'AER'], marker=marker, s=60, edgecolor=color_f[1], facecolor="none", linewidth=2)
            ax[0, r].scatter(x=4, y=sf_df2['sf_best'][sf_df2['forcing'] == 'NAT'], marker=marker, s=60, edgecolor=color_f[2], facecolor="none", linewidth=2)

            ax[0, r].scatter(x=2, y=sf_df2['p_cons'][sf_df2['forcing'] == 'GHG'], marker="1", s=60,
                             color="gray", linewidth=1, zorder = 100)
            ax[0, r].scatter(x=3, y=sf_df2['p_cons'][sf_df2['forcing'] == 'AER'], marker="1", s=60,
                             color="gray", linewidth=1, zorder = 100)
            ax[0, r].scatter(x=4, y=sf_df2['p_cons'][sf_df2['forcing'] == 'NAT'], marker="1", s=60,
                             color="gray", linewidth=1, zorder = 100)

            ax[0, r].set(xlim=(1, 5))
            ax[0, r].set_xticks([])
            ax[0, r].grid(False)
            ax[0, r].set(ylim=(-6, 8))
            y_major_locator = MultipleLocator(2)
            ax[0, r].yaxis.set_major_locator(y_major_locator)

            bwith = 1.5  # 边框宽度设置为2
            ax[0, r].spines['right'].set_linewidth(bwith)
            ax[0, r].spines['left'].set_linewidth(bwith)
            ax[0, r].spines['top'].set_linewidth(bwith)
            ax[0, r].spines['bottom'].set_linewidth(bwith)

            if r > 0:
                ax[0, r].set_yticks([])
            if r == 0:
                ax[0, r].spines['right'].set_linestyle("--")
                ax[0, r].spines['right'].set_linewidth(0.5)
                ax[0, r].spines['right'].set_color("black")
                ax[0, r].spines['right'].set_alpha(0.5)
            if 0 < r < 18:
                ax[0, r].spines['left'].set_visible(False)
                ax[0, r].spines['right'].set_linestyle("--")
                ax[0, r].spines['right'].set_linewidth(0.5)
                ax[0, r].spines['right'].set_color("black")
                ax[0, r].spines['right'].set_alpha(0.5)
            if r == 2:
                ax[0, r].spines['right'].set_linestyle("-")
                ax[0, r].spines['right'].set_linewidth(1)
                ax[0, r].spines['right'].set_color("black")
            if r == 18:
                ax[0, r].spines['left'].set_visible(False)
            if r == 2 or r == 7 or r == 11 or r == 17:
                ax[0, r].spines['right'].set_linestyle("-")
                ax[0, r].spines['right'].set_linewidth(1.5)
                ax[0, r].spines['right'].set_color("black")
                ax[0, r].spines['right'].set_alpha(1)
            ax[0, r].set_xticks([])
            ax[0, r].grid(False)
            ax[0, r].set(ylim=(-6, 8))
            y_major_locator = MultipleLocator(2)
            ax[1, r].yaxis.set_major_locator(y_major_locator)

            #     ############# trend ##############3
            sf_df2 = scaling_factor_and_trends_3sig[scaling_factor_and_trends_3sig['zhishu'] == zhishu]
            sf_df2 = sf_df2[sf_df2['domain'] == domain]
            sf_df2 = sf_df2[sf_df2['jijie'] == jijie]
            err1_GHG = sf_df2['trend'][sf_df2['forcing'] == 'GHG'] - sf_df2['trend_min'][sf_df2['forcing'] == 'GHG']
            err2_GHG = sf_df2['trend_max'][sf_df2['forcing'] == 'GHG'] - sf_df2['trend'][sf_df2['forcing'] == 'GHG']
            err1_aer = sf_df2['trend'][sf_df2['forcing'] == 'AER'] - sf_df2['trend_min'][sf_df2['forcing'] == 'AER']
            err2_aer = sf_df2['trend_max'][sf_df2['forcing'] == 'AER'] - sf_df2['trend'][sf_df2['forcing'] == 'AER']
            err1_nat = sf_df2['trend'][sf_df2['forcing'] == 'NAT'] - sf_df2['trend_min'][sf_df2['forcing'] == 'NAT']
            err2_nat = sf_df2['trend_max'][sf_df2['forcing'] == 'NAT'] - sf_df2['trend'][sf_df2['forcing'] == 'NAT']

            ax[1, r].axhline(y=0, color='black', alpha=0.6, linewidth=0.8)
            p1 = ax[1, r].bar(2, sf_df2['trend'][sf_df2['forcing'] == 'GHG'], yerr=[[err1_GHG.values[0]], [err1_GHG.values[0]]], alpha=1, align='center', edgecolor=color_f[0], facecolor=color_f[0], width=0.8, error_kw=dict(lw=0.4, capsize=0, capthick=0.4))
            p2 = ax[1, r].bar(3, sf_df2['trend'][sf_df2['forcing'] == 'AER'], yerr=[[err1_aer.values[0]], [err1_aer.values[0]]], alpha=1, align='center', edgecolor=color_f[1], facecolor=color_f[1], width=0.8, error_kw=dict(lw=0.4, capsize=0, capthick=0.4))
            p3 = ax[1, r].bar(4, sf_df2['trend'][sf_df2['forcing'] == 'NAT'], yerr=[[err1_nat.values[0]], [err2_nat.values[0]]], alpha=1, align='center', edgecolor=color_f[2], facecolor=color_f[2], width=0.8, error_kw=dict(lw=0.4, capsize=0, capthick=0.4))

            ax[1, r].set(xlim=(1, 5))
            labels = ['GHG', 'AER', 'NAT']
            ax[1, r].set_xlabel(domain.replace("AllMask", "NHL").replace("WetRegion", "HR").replace("DryRegion", "LR"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 15},)
            ax[1, r].set_xticks([])
            ax[1, r].grid(False)
            ax[1, r].set(ylim=(-2.5, 4.5))
            ax[1, r].yaxis.set_major_locator(y_major_locator)
            bwith = 1.5  # 边框宽度设置为2
            ax[1, r].spines['right'].set_linewidth(bwith)
            ax[1, r].spines['left'].set_linewidth(bwith)
            ax[1, r].spines['top'].set_linewidth(bwith)
            ax[1, r].spines['bottom'].set_linewidth(bwith)
            if r > 0:
                ax[1, r].set_yticks([])
            if r == 0:
                ax[1, r].spines['right'].set_linestyle("--")
                ax[1, r].spines['right'].set_linewidth(0.5)
                ax[1, r].spines['right'].set_color("black")
                ax[1, r].spines['right'].set_alpha(0.5)
            if 0 < r < 18:
                ax[1, r].spines['left'].set_visible(False)
                ax[1, r].spines['right'].set_linestyle("--")
                ax[1, r].spines['right'].set_linewidth(0.5)
                ax[1, r].spines['right'].set_color("black")
                ax[1, r].spines['right'].set_alpha(0.5)
            if r == 2:
                ax[1, r].spines['right'].set_linestyle("-")
                ax[1, r].spines['right'].set_linewidth(1)
                ax[1, r].spines['right'].set_color("black")
            if r == 18:
                ax[1, r].spines['left'].set_visible(False)
            if r == 2 or r == 7 or r == 11 or r == 17:
                ax[1, r].spines['right'].set_linestyle("-")
                ax[1, r].spines['right'].set_linewidth(1.5)
                ax[1, r].spines['right'].set_color("black")
                ax[1, r].spines['right'].set_alpha(1)

        labels = ['GHG', 'AER', 'NAT', 'GHG', 'AER', 'NAT']
        legend_elements = [
            Line2D([0], [0], marker=marker, color=color_f[0], markeredgecolor=color_f[0], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[1], markeredgecolor=color_f[1], markerfacecolor="none", markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[2], markeredgecolor=color_f[2], markerfacecolor="none", markersize=8, linewidth=2),
            Patch(facecolor=color_f[0], edgecolor=color_f[0], alpha=1),
            Patch(facecolor=color_f[1], edgecolor=color_f[1], alpha=1),
            Patch(facecolor=color_f[2], edgecolor=color_f[2], alpha=1),]
        handles = legend_elements
        anchor_x = -0.5
        anchor_y = 0

        ax[1, 1].legend(handles, labels, frameon=False, ncol=6, fontsize=15, fancybox=True, loc='upper right',
                        bbox_to_anchor=(15, -0.1))  # bbox_to_anchor=[1.01,0.7],
        ax[0, 0].text(0, 1.2, jijie.replace("ANN", " (a) Three Signal - ANN").replace("JJA", " (c) Three Signal - JJA").replace("DJF", " (e) Three Signal - DJF").replace("SON", " (d) Three Signal - SON").replace("MAM", " (b) Three Signal - MAM"),
                      fontdict={'family': 'Times New Roman', 'fontsize': 25, 'weight': 'bold'},
                      transform=ax[0, 0].transAxes)
        ax[0, 0].set_title(' Scaling Factor', font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22},
                           loc="left", pad=0.01)
        ax[1, 0].set_title(' Trend (%/decade)', font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22},
                           loc="left", pad=0.01)
        plt.tight_layout()
        fig_dir = 'C:/5_attribution/res/pic'
        # fig.savefig(fig_dir + 'Fig4_detection_and_attribution_3singal&trend_' + zhishu + '_' + jijie + '.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()
