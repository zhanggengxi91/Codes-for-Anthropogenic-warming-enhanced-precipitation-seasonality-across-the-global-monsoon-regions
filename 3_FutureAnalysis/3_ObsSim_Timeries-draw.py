import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cftime
import pandas as pd

def read_data(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith('.nc'):
            common_time = xr.cftime_range(start='1950-07-01', end='2014-07-01', freq='AS-JUL', calendar='noleap')
            nc_file = xr.open_dataset(os.path.join(folder, file))
            nc_file = nc_file.assign_coords(time=common_time)
            filename = file.split("_")[3]
            data.append(nc_file)
    return data


def calculate_spatial_mean(data):
    return [ds['SI'].mean(dim=('lat', 'lon'), skipna=True) for ds in data]


def calculate_anomalies(time_series_data):
    return [ts - ts.mean() for ts in time_series_data]

def percentile_range(simulated_data, q1=10, q2=90):
    concat_data = xr.concat(simulated_data, dim='ensemble')
    return concat_data.quantile(q1 / 100, dim='ensemble'), concat_data.quantile(q2 / 100, dim='ensemble')



observed_data = read_data('G:/2_Monsoon_Distribution-Trend/2_Monsoon_Timeseries/5_Drawtimeries/SI_observed')
simulated_data = read_data('G:/2_Monsoon_Distribution-Trend/2_Monsoon_Timeseries/5_Drawtimeries/SI_historical')

observed_spatial_mean = calculate_spatial_mean(observed_data)
simulated_spatial_mean = calculate_spatial_mean(simulated_data)

observed_anomalies = calculate_anomalies(observed_spatial_mean)
simulated_anomalies = calculate_anomalies(simulated_spatial_mean)
mean_simulated_anomalies = sum(simulated_anomalies) / len(simulated_anomalies)
print(mean_simulated_anomalies)

lower_bound, upper_bound = percentile_range(simulated_anomalies)

# Plotting
plt.rc('font', family='Times New Roman', size=40)
fig, ax1 = plt.subplots(figsize=(21, 7), dpi=100)
for i, ts in enumerate(observed_anomalies):
    ts['time'] = pd.to_datetime(ts['time'].astype(str))
    plt.plot(ts['time'], ts, label=f'Observed {i+1}', linewidth=3)
    ax1.set(ylim=(-0.01, 0.01))

plt.plot(mean_simulated_anomalies['time'], mean_simulated_anomalies, color="black", label='Mean Simulated', linewidth=3, linestyle='-', zorder = 100)
plt.fill_between(lower_bound['time'], lower_bound, upper_bound, color="gray", alpha=0.2)
plt.title('     SI Anomalies ', font={'family': 'Times New Roman', 'weight': 'bold', 'size': 45}, loc="left", pad=20)


bwith = 3  # 边框宽度设置为2
ax1.spines['bottom'].set_linewidth(bwith)  # 图框下边
ax1.spines['left'].set_linewidth(bwith)  # 图框左边
ax1.spines['top'].set_linewidth(bwith)  # 图框上边
ax1.spines['right'].set_linewidth(bwith)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40

plt.tick_params(top=False, bottom=True, left=True, right=False)
plt.tick_params(axis="x", direction='in', which="major", length=10, width=3)
plt.tick_params(axis="y", direction='in', which="major", length=12.5, width=2)

# ax1.set(ylim=(-0.01, 0.01))
# ax1.set_yticks([-0.01, 0, 0.01])
# ax1.set(xlim=(1950, 2015))
# ax1.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010])
# ax1.set_xticklabels([1950, 1960, 1970, 1980, 1990, 2000, 2010])

# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#            fancybox=True, shadow=True, ncol=5, prop={'size': 40, "family": "Times New Roman"})
plt.savefig('G:/2_Monsoon_Distribution-Trend/2_Monsoon_Timeseries/5_Drawtimeries/global_mean_SI_anomalies.png', dpi=300,
            format='png', transparent=True, bbox_inches='tight', pad_inches=0.2)
plt.show()


# print("simulated_anomalies shape:", observed_anomalies.shape)
# print("simulated_anomalies dims:", observed_anomalies.dims)




