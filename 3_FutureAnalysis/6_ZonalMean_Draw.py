import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the data folder and the observed and model subfolders
data_folder = 'G:/2_Monsoon_Distribution-Trend/1_monsoon_SpatialDistribution/2_ZonalMean'
subfolders = ['pr', 'AE', 'SI']

# Define the variable names corresponding to each subfolder
observed_variable_names = ['pr', 'pr', 'SI']
historical_variable_names = ['pr', 'pr', 'SI']

# Preallocate lists for the results
observed_means = []
model_means = []
model_10p = []
model_90p = []

# Process each subfolder
for sub, obs_var, hist_var in zip(subfolders, observed_variable_names, historical_variable_names):
    # Open all observed datasets, calculate the time mean, then the zonal mean
    observed_files = os.listdir(os.path.join(data_folder, sub, 'observed'))
    observed_data = xr.concat([xr.open_dataset(os.path.join(data_folder, sub, 'observed', file))[obs_var].mean(dim='time') for file in observed_files], dim='obs')
    observed_zonal_mean = observed_data.mean(dim='lon')
    observed_means.append(observed_zonal_mean)

    # Open all model datasets, calculate the time mean, then the zonal mean
    model_files = os.listdir(os.path.join(data_folder, sub, 'historical'))
    model_data = xr.concat([xr.open_dataset(os.path.join(data_folder, sub, 'historical', file))[hist_var].mean(dim='time') for file in model_files], dim='model')
    model_zonal_mean = model_data.mean(dim='lon')
    try:
        model_means.append(model_zonal_mean.mean(dim='model'))  # CMIP6 ensemble mean
        model_10p.append(model_zonal_mean.quantile(0.1, dim='model'))  # 10% threshold
        model_90p.append(model_zonal_mean.quantile(0.9, dim='model'))  # 90% threshold
    except RuntimeWarning:
        # All-NaN slice encountered
        pass


def zonal_plot(data, lat, label, ax, sub, color=None):
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 40}
    ax.plot(data, lat, linewidth=3, label=label, color=color)
    ax.tick_params(top=False, bottom=True, left=True, right=False)
    ax.tick_params(axis="x", direction='in', which="major", length=10, width=3)
    ax.tick_params(axis="y", direction='in', which="major", length=12.5, width=2)
    bwith = 4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.set_yticks([-60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(["60$^°$S", "30$^°$S", "EQ", "30$^°$N", "60$^°$N", "90$^°$N"], fontdict=font)
    ax.tick_params(labelsize=40)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_ylim(-60, 90)


# Define xlim dictionary for each subfolder
xlim_dict = {'SI': (0, 0.15), 'AE': (0, 2.0), 'pr': (0, 3000)}  # replace 'other_folder' with actual folder names, xmin, xmax with actual values

# Plot the results
for i, sub in enumerate(subfolders):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("        " + sub.replace("AE", "RE").replace("pr", "ATP"), font={'family': 'Times New Roman', 'weight': 'bold', 'size': 45},
                 loc="left", pad=20)

    # Check if 'obs' dimension is present
    if 'obs' in observed_means[i].dims:
        # Determine the number of observed lines
        num_obs = len(observed_means[i]['obs'])

        # Create a colormap for observed lines
        obs_cmap = plt.get_cmap('Set1')

        # Iterate over the observed lines
        for obs in range(num_obs):
            legend_label = observed_files[obs].split("_")[2]
            # Get the color for the observed line based on the colormap
            obs_color = obs_cmap(obs % 10)
            # Call the zonal_plot function with the observed line color
            zonal_plot(observed_means[i].sel(obs=obs), observed_means[i]['lat'], legend_label, ax, sub, color=obs_color)
    else:
        zonal_plot(observed_means[i], observed_means[i]['lat'], 'Observed', ax, sub)

    zonal_plot(model_means[i], model_means[i]['lat'], 'CMIP6', ax, sub, color='black')
    ax.fill_betweenx(model_means[i]['lat'], model_10p[i], model_90p[i], alpha=0.12, label='CMIP6 10-90 percentile',
                     color='black')

    # Set xlim based on the subfolder
    if sub in xlim_dict:
        ax.set_xlim(xlim_dict[sub])
        # If subfolder is 'SI', set the xticks and xticklabels
        if sub == 'SI':
            ax.set_xticks([0, 0.05, 0.1, 0.15])
            ax.set_xticklabels(['0','0.5', '1.0', '1.5'])
        elif sub == "AE":

            ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
            ax.set_xticklabels(['0',"0.5", "1.0", "1.5", "2.0"])
        elif sub == "SI":

            ax.set_xticks([0, 0.05, 0.10, 0.15])
            ax.set_xticklabels(['0',"0.05", "0.10", "0.15"])
    else:
        ax.set_xlim([np.min(model_10p[i]), np.max(model_90p[i])])

    plt.savefig(f"G:/2_Monsoon_Distribution-Trend/1_monsoon_SpatialDistribution/Land-ZonalMean_{sub}_Legend.png",
                format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.show()
