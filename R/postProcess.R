library(terra)
library(dplyr)
library(data.table)
library(sf)
library(reproducible)

# First getting study area
studyArea <- reproducible::prepInputs(url = "https://sis.agr.gc.ca/cansis/nsdb/ecostrat/district/ecodistrict_shp.zip",
                                 destinationPath = '~/.')
studyArea <- studyArea[studyArea$ECOREGION %in% c("176", "175"),]
studyArea <- sf::st_transform(studyArea, "EPSG:3579")
studyArea <- terra::aggregate(terra::vect(studyArea))

# Getting input covariates in order to follow stand type and stand age
df <- data.table::fread("../../inputsStack.csv")
df <- df |>
  dplyr::select(x, y, Type, FIRE_YEAR)

# Cropping input covariates to the size of the study Area
pts <- st_as_sf(df, coords = c("x", "y"), crs = 3579) |> terra::vect() |> reproducible::Cache()
studyArea_ <- terra::intersect(studyArea, pts) |> reproducible::Cache()

getSCFMRast <- function(year) {
  burnMap <- terra::rast(paste0("cumulativeBurnMap_year_", year, "test.tif"))
  return(burnMap)
}

# Load cumulative burn map from SCFM
burnMap <- getSCFMRast(2034)

# Check overlap: extract raster values per polygon
vals <- terra::extract(burnMap, studyArea_, fun = max, na.rm = TRUE)

# vals has ID + burnMap value(s)
# polygons with any burned pixel will have burnMap > 0
burned <- vals$PolyID > 0

# update FIRE_YEAR
studyArea_$FIRE_YEAR[burned] <- 0

