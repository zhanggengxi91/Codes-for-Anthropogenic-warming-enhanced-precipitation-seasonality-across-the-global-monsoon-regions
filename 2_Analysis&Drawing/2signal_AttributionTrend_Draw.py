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
scaling_factor_and_trends_2sig = pd.read_csv('C:/5_attribution/res/trends_scaling_factors_2signal_GPH.csv', index_col=0)
# ####################   transfer 5 year to decade   #################
scaling_factor_and_trends_2sig['trend'] = scaling_factor_and_trends_2sig['trend'] * 200
scaling_factor_and_trends_2sig['trend_min'] = scaling_factor_and_trends_2sig['trend_min'] * 200
scaling_factor_and_trends_2sig['trend_max'] = scaling_factor_and_trends_2sig['trend_max'] * 200

plt.rc('font', family='Times New Roman', size=18)

for zhishu in ['rx1day', 'rx5day']:
    for jijie in ['ANN', 'MAM', 'JJA', 'SON', 'DJF']:
        fig, ax = plt.subplots(2, 19, sharex=False, figsize=(16, 4), gridspec_kw={'height_ratios': [2, 1], 'wspace': 0, 'hspace': 0.25})
        regions = ['AllMask', 'DryRegion', 'WetRegion', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'NEU', 'WCE', 'MED', 'EEU','WSB', 'ESB', 'RFE', 'WCA', 'EAS', 'SAS', 'WAF']
        # color_f = ['#bcbddc', '#a1d99b']
        color_f = ['blue','red']

        # ####################   draw scaling factor   #################
        for r in range(len(regions)):
            marker = 'o'
            domain = regions[r]
            sf_df = scaling_factor_and_trends_2sig[scaling_factor_and_trends_2sig['zhishu'] == zhishu]
            sf_df = sf_df[sf_df['domain'] == domain]
            sf_df = sf_df[sf_df['jijie'] == jijie]
            ax[0, r].axhline(y=0, color='black', alpha=0.6, linewidth=1.5)
            ax[0, r].axhline(y=1, linestyle=':', color='black', alpha=0.6, linewidth=1.5)
            ax[0, r].plot((1, 1), (sf_df['sf_min'][sf_df['forcing'] == 'ANT'], sf_df['sf_max'][sf_df['forcing'] == 'ANT']), color=color_f[0], alpha=1, linewidth=2)
            ax[0, r].plot((2, 2), (sf_df['sf_min'][sf_df['forcing'] == 'NAT'], sf_df['sf_max'][sf_df['forcing'] == 'NAT']), color=color_f[1], alpha=1, linewidth=2)
            ax[0, r].scatter(x=1, y=sf_df['sf_best'][sf_df['forcing'] == 'ANT'], marker=marker, s=60, edgecolor=color_f[0], facecolor="none", linewidth=2)
            ax[0, r].scatter(x=2, y=sf_df['sf_best'][sf_df['forcing'] == 'NAT'], marker=marker, s=60, edgecolor=color_f[1], facecolor="none", linewidth=2)
            print(sf_df['sf_min'][sf_df['forcing'] == 'ANT'])

            ax[0, r].scatter(x=1, y=sf_df['p_cons'][sf_df['forcing'] == 'ANT'], marker="1", s=60, edgecolor="gray", facecolor="gray",
                             linewidth=1, zorder = 100)
            ax[0, r].scatter(x=2, y=sf_df['p_cons'][sf_df['forcing'] == 'NAT'], marker="1", s=60, edgecolor="gray", facecolor="gray",
                             linewidth=1, zorder = 100)

            ax[0, r].set(xlim=(0, 3))
            ax[0, r].set_xticks([])
            ax[0, r].set_yticks([])
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

            #     ############# trend ##############3
            sf_df2 = scaling_factor_and_trends_2sig[scaling_factor_and_trends_2sig['zhishu'] == zhishu]
            sf_df2 = sf_df2[sf_df2['domain'] == domain]
            sf_df2 = sf_df2[sf_df2['jijie'] == jijie]
            err1_ant = sf_df2['trend'][sf_df2['forcing'] == 'ANT'] - sf_df2['trend_min'][sf_df2['forcing'] == 'ANT']
            err2_ant = sf_df2['trend_max'][sf_df2['forcing'] == 'ANT'] - sf_df2['trend'][sf_df2['forcing'] == 'ANT']
            err1_nat = sf_df2['trend'][sf_df2['forcing'] == 'NAT'] - sf_df2['trend_min'][sf_df2['forcing'] == 'NAT']
            err2_nat = sf_df2['trend_max'][sf_df2['forcing'] == 'NAT'] - sf_df2['trend'][sf_df2['forcing'] == 'NAT']
            ax[1, r].axhline(y=0, color='black', alpha=0.6, linewidth=0.8)
            p1 = ax[1, r].bar(1, sf_df2['trend'][sf_df2['forcing'] == 'ANT'], yerr=[[err1_ant.values[0]], [err1_ant.values[0]]], alpha=1, align='center', edgecolor=color_f[0], facecolor=color_f[0], width=0.8, error_kw=dict(lw=0.5, capsize=0, capthick=0.5))
            p2 = ax[1, r].bar(2, sf_df2['trend'][sf_df2['forcing'] == 'NAT'], yerr=[[err1_nat.values[0]], [err2_nat.values[0]]], alpha=1, align='center', edgecolor=color_f[1], facecolor=color_f[1], width=0.8, error_kw=dict(lw=0.5, capsize=0, capthick=0.5))
            ax[1, r].set(xlim=(0, 3))
            labels = ['ANT', 'NAT']
            ax[1, r].set_xlabel(domain.replace("AllMask", "NHL").replace("WetRegion", "HR").replace("DryRegion", "LR"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 15},)
            ax[1, r].set_xticks([])
            ax[1, r].set_yticks([])
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






        labels = ['ANT', 'NAT', 'ANT', 'NAT']
        legend_elements = [
            Line2D([0], [0], marker=marker, color=color_f[0], label='location', markeredgecolor=color_f[0], markerfacecolor="none",markersize=8, linewidth=2),
            Line2D([0], [0], marker=marker, color=color_f[1], label='location', markeredgecolor=color_f[1], markerfacecolor="none",markersize=8, linewidth=2),
            Patch(facecolor=color_f[0], edgecolor=color_f[0], label='ANT', alpha=1),
            Patch(facecolor=color_f[1], edgecolor=color_f[1], label='NAT', alpha=1)]
        handles = legend_elements
        ax[1, 1].legend(handles, labels, frameon=False, ncol=4, fontsize=15, fancybox=True, loc='upper right', bbox_to_anchor=(13, -0.1))  # bbox_to_anchor=[1.01,0.7],
        ax[0, 0].text(0, 1.2, jijie.replace("ANN", " (a) Two Signal - ANN").replace("JJA", " (c) Two Signal - JJA").replace("DJF", " (e) Two Signal - DJF").replace("SON", " (d) Two Signal - SON").replace("MAM", " (b) Two Signal - MAM"),
                      fontdict={'family': 'Times New Roman', 'fontsize': 25, 'weight': 'bold'},
                      transform=ax[0, 0].transAxes)
        ax[0, 0].set_title(' Scaling Factor',font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, loc = "left", pad=0.01)
        ax[1, 0].set_title(' Trend (%/decade)',font={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, loc = "left", pad=0.01)
        anchor_x = -0.5
        anchor_y = 0


        plt.tight_layout()
        fig_dir = 'C:/5_attribution/res/pic'
        # fig.savefig(fig_dir + 'Fig4_detection_and_attribution_2singal&trend_' + zhishu + '_' + jijie + '.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()
