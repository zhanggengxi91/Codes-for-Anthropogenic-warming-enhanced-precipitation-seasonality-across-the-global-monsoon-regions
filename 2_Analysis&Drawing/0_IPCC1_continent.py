import geopandas as gpd
import matplotlib.pyplot as plt

#Importing an ESRI Shapefile and plotting it using GeoPandas
reference_region=gpd.read_file("F:/PR_attribution3_indices/IPCC_WetDry/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
# print(reference_region.shape)
# print(reference_region.head())
# where jet comes from----matplotlib library
# reference_region.plot(facecolor="none",edgecolor='black',column='Continent')
# plt.show()


continental_region = gpd.read_file("F:/PR_attribution3_indices/IPCC_WetDry/IPCC-WGII-continental-regions_shapefile/IPCC_WGII_continental_regions.shp")
print(continental_region.shape)
print(continental_region.head())
continental_region.plot(facecolor='none', edgecolor='black', column='Region')
plt.show()

NorthAmerica = continental_region[continental_region['Region'] == "North America"]
NorthAmerica.plot(cmap='jet',edgecolor='black')
NorthAmerica.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/NorthAmerica.shp', driver="ESRI Shapefile")

# plt.show()

Europe = continental_region[continental_region['Region'] == "Europe"]
Europe.plot(cmap='jet',edgecolor='black')
Europe.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/Europe.shp', driver="ESRI Shapefile")
# plt.show()

Asia = continental_region[continental_region['Region'] == "Asia"]
Asia.plot(cmap='jet',edgecolor='black')
Asia.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/Asia.shp', driver="ESRI Shapefile")
# plt.show()

Africa = continental_region[continental_region['Region'] == "Africa"]
Africa.plot(cmap='jet',edgecolor='black')
Africa.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/Africa.shp', driver="ESRI Shapefile")
# plt.show()

Australasia = continental_region[continental_region['Region'] == "Australasia"]
Australasia.plot(cmap='jet',edgecolor='black')
Australasia.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/Australasia.shp', driver="ESRI Shapefile")
# plt.show()

SouthAmerica = continental_region[continental_region['Region'] == "Central and South America"]
SouthAmerica.plot(cmap='jet',edgecolor='black')
SouthAmerica.to_file('G:/1_OrigionalData/0_DefineMonsoonRegions/SouthAmerica.shp', driver="ESRI Shapefile")
# plt.show()

