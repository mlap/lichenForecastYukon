library(terra)
library(sf)
library(dplyr)
library(ggplot2)
library(leaflet)
library(caret)
library(data.table)
library(rnaturalearth)
library(rnaturalearthdata)
library(whitebox)
library(tictoc)

prepCovariates <- function(){
  # Getting yukon SF object to build a template from
  canada <- ne_states(country = "canada", returnclass = "sf")
  yukon <- canada |> filter(name_en == "Yukon")
  yukon <- st_transform(yukon, "EPSG:3579")
  
  # Creating a spatVect of the yukon Object and rasterizing it with 480 cells
  yukonVect <- terra::vect(yukon)
  yukon_raster_template <- rast(ext(yukonVect), resolution = 240, crs = crs(yukonVect))
  yukon_raster <- rasterize(yukonVect, yukon_raster_template, field = 0)
  rm(yukon, canada)
  
  # Getting Fire history and converting it to a spatRaster
  fireHistory <- st_read("./data/data/Fire_History.shp")
  fireHistoryExt <- fireHistory %>% filter(FIRE_YEAR > 3000) %>% mutate(FIRE_YEAR = DECADE)
  fireHistoryEdit <- fireHistory %>% filter(FIRE_YEAR < 3000)
  fireHistory <- rbind(fireHistoryEdit, fireHistoryExt)
  fireVect <- terra::project(vect(fireHistory), crs(yukon_raster))
  fireRast <- terra::rasterize(fireVect, yukon_raster, field="FIRE_YEAR")
  tSinceFireRast <- 2025 - fireRast
  tSinceFireRast <- ceiling(tSinceFireRast / 10) * 10 
  rm(fireHistoryEdit, fireHistoryExt, fireHistory, fireVect, fireRast)
  
  # Getting topographic indices of slope, elev, aspect
  demFile <- "./data/data/50n150w_20101117_gmted_med075.tif"
  dem <- rast(demFile)
  aspect <- terra::terrain(dem, v = "aspect", unit = "radians")
  
  # Calculating topographic wetness index
  # (Skipping some lines as whitebox outputs tifs that can be reloaded)
  # wbt_breach_depressions(demFile, "./filledDEM.tif")
  # wbt_slope("./filledDEM.tif", "./slopeDEM.tif", units = "radians")
  # wbt_d8_flow_accumulation("./filledDEM.tif", "./d8flowDEM.tif")
  # wbt_depth_to_water("./filledDEM.tif", "./dtwDEM.tif") # Doesn't work due to propreitary licensing
  slope <- rast("./slopeDEM.tif")
  d8flow <- rast("./d8flowDEM.tif")
  wbt_twi <- log(d8flow / (tan(slope) + 0.001))
  
  # Getting tree cover data
  # Loading NTEMS data and checking against at-site classifications
  vegetationInventory <- terra::rast("./data/data/CA_Forest_Tree_Species_2022.tif")
  yukon_box <- terra::ext(-141, -123, 59.5, 69.5)  # xmin, xmax, ymin, ymax
  yukon_poly_ll <- as.polygons(yukon_box, crs = "EPSG:4326")
  yukon_poly_proj <- project(yukon_poly_ll, crs(vegetationInventory))
  
  # Cropping and projecting NTEMS sf object
  vegInventory_cropped <- terra::crop(vegetationInventory, yukon_poly_proj)
  vegInventory_cropped <- terra::project(vegInventory_cropped, crs(yukon_raster))
  vegInventoryYukon <- terra::mask(vegInventory_cropped, yukonVect)
  rm(yukon_poly_ll, yukon_poly_proj, vegInventory_cropped)
  
  # Linking NTEMS id's to stand type
  speciesKey <- read.csv("./data/data/sppEquivalencies_CA.csv") |> 
    filter(is.na(NTEMS_Species_Code) == FALSE)
  vegLegend <- speciesKey[, c("NTEMS_Species_Code", "Type")] |> unique()
  vegLegend[nrow(vegLegend) + 1, ] <- c(0, "Unf")
  abbreviationLabels <- c("C", "B", "U")
  abbreviationCodes <- c("Conifer", "Deciduous", "Unf")
  vegLegend$Type <- abbreviationLabels[match(vegLegend$Type, abbreviationCodes)]
  vegLegend$NTEMS_Species_Code <- as.numeric(vegLegend$NTEMS_Species_Code)
  levels(vegInventoryYukon) <- vegLegend
  rm(speciesKey, vegLegend, abbreviationLabels, abbreviationCodes)
  
  # Filling in background of fire year and Stand type
  fireBackground <- rasterize(yukonVect, yukon_raster_template, field = 0)
  yukonVect$zone <- "U"
  vegBackground <- rasterize(yukonVect, yukon_raster_template, field = "zone")
  tSinceFireRastFilled <- terra::merge(tSinceFireRast, fireBackground)
  vegInventoryYukonFilled <- terra::merge(vegInventoryYukon, vegBackground)
  rm(fireBackground, vegBackground, tSinceFireRast, vegInventoryYukon)
  
  # Project to the same CRS
  slope <- terra::project(slope, crs(yukon_raster))
  elev <- terra::project(dem, crs(yukon_raster))
  aspect <- terra::project(aspect, crs(yukon_raster))
  wbt_twi <- terra::project(wbt_twi, crs(yukon_raster))
  
  lichen_presence_absence <- rast("ensemble_prediction.tif")
  lichen_presence_absence <- terra::mask(lichen_presence_absence, yukonVect)
  
  # Resampling points so that everything has same cell geometry
  elev_rs <- terra::resample(terra::mask(elev, yukonVect), yukon_raster, "average")
  slope_rs <- terra::resample(slope, yukon_raster, "average")
  aspect_rs <- terra::resample(aspect, yukon_raster, "average")
  wbt_twi_rs <- terra::resample(wbt_twi, yukon_raster, "average")
  vegInventoryYukon_rs <- terra::resample(vegInventoryYukonFilled, yukon_raster, "mode")
  tSinceFireRast_rs <- terra::resample(tSinceFireRastFilled, yukon_raster, "mode")
  lichen_presence_absence_rs <- terra::resample(lichen_presence_absence, yukon_raster, "max")
  rm(slope, elev, aspect, wbt_twi, vegInventoryYukonFilled, tSinceFireRastFilled, lichen_presence_absence)
  
  # Collating all the covariates and predictors
  inputs_stack <- c(slope_rs, elev_rs, aspect_rs, wbt_twi_rs, tSinceFireRast_rs, vegInventoryYukon_rs, lichen_presence_absence_rs) # Also add time since fire and stand classification
  inputsDT <- as.data.table(inputs_stack, na.rm = TRUE, xy = TRUE)
  data.table::fwrite(inputsDT, file = "inputsStack.csv", row.names = FALSE)
}

  