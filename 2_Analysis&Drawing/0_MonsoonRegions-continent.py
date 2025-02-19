# -- coding:utf-8 --
import geopandas as gpd
import xarray as xr
import regionmask

# Load the shapefile containing country boundaries and continent information
shapefile_path = "G:/1_OrigionalData/0_DefibeMonsoonRegions/Countries/ne_50m_admin_0_countries.shp"
gdf = gpd.read_file(shapefile_path)
print(gdf)
# Define the continents
continents = {
    "Asia": "Asia",
    "Africa": "Africa",
    "Europe": "Europe",
    "North America": "North America",
    "South America": "South America",
    "Australia": "Australia",
    "Antarctica": "Antarctica",
}

# Load the global netCDF file
ds = xr.open_dataset("G:/1_OrigionalData/0_DefibeMonsoonRegions/CRU_monsoon_regions_ready.nc")

# Create masks for each continent and save the subregion netCDF file
for continent_name, continent_code in continents.items():
    # Extract the continent geometries from the GeoDataFrame
    continent_gdf = gdf[gdf["CONTINENT"] == continent_name]

    # Create the mask for the current continent
    mask = regionmask.mask_geopandas(continent_gdf, ds)

    # Apply the mask to the original netCDF dataset
    subregion_ds = ds.where(mask.notnull())

    # Save the subregion dataset as a netCDF file
    subregion_ds.to_netcdf(f"{continent_name}_data.nc")
