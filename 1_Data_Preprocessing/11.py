
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pymannkendall as  trend_mk_test

# Load netCDF file
nc_file = 'G:/1_OrigionalData/4_AE_SpatialTrend/4_ChangePoint/1/AENew-land_yr_ERA5_observed_runmean_1950-2014_TimeSeries.nc'

# Extract time, lat, lon, and variable data
time_var = nc_file.time['time']
print(time_var)
time = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
lat = nc_file.lat['lat'][:]
lon = nc_file.lon['lon'][:]
variable = nc_file.pr['pr'][:]

# Initialize arrays to store change point information
change_years = np.zeros((lat.shape[0], lon.shape[0]))
change_years[:] = np.nan

# Loop over each location in the lat-lon grid and apply the Mann-Kendall test
# Loop over each location in the lat-lon grid and apply the Mann-Kendall test
for i in range(lat.shape[0]):
    for j in range(lon.shape[0]):
        # Extract time series at current location
        ts = variable[:, i, j]

        # Apply Mann-Kendall test
        trend, h, p, z, Tau, s, var_s, slope, intercept = trend_mk_test(ts)

        # Determine change points and corresponding years
        change_points = np.where(h == 1)[0]
        if len(change_points) > 0:
            change_years[i, j] = time[change_points[0]].year

# Plot results using a scatter plot
lon_2d, lat_2d = np.meshgrid(lon, lat)
plt.scatter(lon_2d.flatten(), lat_2d.flatten(), c=change_years.flatten())
plt.colorbar(label='Year of Change Point')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
