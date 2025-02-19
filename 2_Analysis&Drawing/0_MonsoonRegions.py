import numpy as np
import xarray as xr

# Read the netCDF file
ds = xr.open_dataset("G:/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_1981-2010Mean.nc")

# Calculate the climatological mean for each month
clim_mean = ds.groupby("time.month").mean("time")

# Define local summer and winter for each hemisphere
nh_summer = clim_mean.sel(month=slice(2, 6))  # Northern Hemisphere: February to June
sh_summer = clim_mean.sel(month=slice(8, 12))  # Southern Hemisphere: August to December

nh_winter = clim_mean.sel(month=slice(8, 12))  # Northern Hemisphere: August to December
sh_winter = clim_mean.sel(month=slice(2, 6))  # Southern Hemisphere: February to June

# Calculate the summer and winter rainfall rates for each hemisphere
nh_summer_mean = nh_summer.mean(dim="month")
nh_winter_mean = nh_winter.mean(dim="month")

sh_summer_mean = sh_summer.mean(dim="month")
sh_winter_mean = sh_winter.mean(dim="month")

# Apply the criteria to identify the monsoon regions
nh_monsoon = ((nh_summer_mean - nh_winter_mean) / 30.44 > 2.0) & (nh_summer_mean / clim_mean.mean(dim="month") > 0.55)
sh_monsoon = ((sh_summer_mean - sh_winter_mean) / 30.44 > 2.0) & (sh_summer_mean / clim_mean.mean(dim="month") > 0.55)

nh_monsoon = nh_monsoon.where((ds.lat >= 0) & (ds.lat <= 90), 0)
sh_monsoon = sh_monsoon.where((ds.lat >= -60) & (ds.lat < 0), 0)

# Combine the monsoon regions for both hemispheres
monsoon_regions = xr.where(nh_monsoon | sh_monsoon, 1, 0)

# Write the output to a new netCDF file
monsoon_regions.to_netcdf("G:/1_OrigionalData/0_DefibeMonsoonRegions/monsoon_regions.nc")


'''
cdo  -selmon,5/9 /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_1981-2010Mean.nc /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_5-9.nc
cdo -selmon,1,2,3,11,12, /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_1981-2010Mean.nc /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_11-3.nc

cdo timsum /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_5-9.nc /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_5-9sum.nc
cdo timsum /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_11-3.nc /mnt/g/1_OrigionalData/0_DefibeMonsoonRegions/pr-land_Amon_CRU_observed_r1i1p1f1_11-3sum.nc
'''