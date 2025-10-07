## Everything in this file and any files in the R directory are sourced during `simInit()`;
## all functions and objects are put into the `simList`.
## To use objects, use `sim$xxx` (they are globally available to all modules).
## Functions can be used inside any function that was sourced in this module;
## they are namespaced to the module, just like functions in R packages.
## If exact location is required, functions will be: `sim$.mods$<moduleName>$FunctionName`.
defineModule(sim, list(
  name = "lichenForecastYukon",
  description = "",
  keywords = "",
  authors = structure(list(list(given = c("Marcus", "Francois"), family = "Lapeyrolerie", role = c("aut", "cre"), email = "mlapeyro@mail.ubc.ca", comment = NULL)), class = "person"),
  childModules = character(0),
  version = list(lichenForecastYukon = "0.0.0.9000"),
  timeframe = as.POSIXlt(c(NA, NA)),
  timeunit = "year",
  citation = list("citation.bib"),
  documentation = list("NEWS.md", "README.md", "lichenForecastYukon.Rmd"),
  reqdPkgs = list("SpaDES.core (>= 2.1.5.9003)", "ggplot2", "reticulate", 
                  "whitebox"),
  parameters = bindrows(
    #defineParameter("paramName", "paramClass", value, min, max, "parameter description"),
    defineParameter("predictionInterval", "numeric", 10, NA, NA, 
                    "Time between predictions"),
    defineParameter(".plots", "character", "screen", NA, NA,
                    "Used by Plots function, which can be optionally used here"),
    defineParameter(".plotInitialTime", "numeric", start(sim), NA, NA,
                    "Describes the simulation time at which the first plot event should occur."),
    defineParameter(".plotInterval", "numeric", NA, NA, NA,
                    "Describes the simulation time interval between plot events."),
    defineParameter(".saveInitialTime", "numeric", NA, NA, NA,
                    "Describes the simulation time at which the first save event should occur."),
    defineParameter(".saveInterval", "numeric", NA, NA, NA,
                    "This describes the simulation time interval between save events."),
    defineParameter(".studyAreaName", "character", NA, NA, NA,
                    "Human-readable name for the study area used - e.g., a hash of the study",
                          "area obtained using `reproducible::studyAreaName()`"),
    ## .seed is optional: `list('init' = 123)` will `set.seed(123)` for the `init` event only.
    defineParameter(".seed", "list", list(), NA, NA,
                    "Named list of seeds to use for each event (names)."),
    defineParameter(".useCache", "logical", FALSE, NA, NA,
                    "Should caching of events or module be used?")
  ),
  inputObjects = bindrows(
    #expectsInput("objectName", "objectClass", "input object description", sourceURL, ...),
    expectsInput(objectName = NA, objectClass = NA, desc = NA, sourceURL = NA)
  ),
  outputObjects = bindrows(
    #createsOutput("objectName", "objectClass", "output object description", ...),
    createsOutput(objectName = NA, objectClass = NA, desc = NA)
  )
))

doEvent.lichenForecastYukon = function(sim, eventTime, eventType) {
  switch(
    eventType,
    init = {
      browser()
      
      # do stuff for this event
      sim <- Init(sim)
      
      # schedule future event(s)
      sim <- scheduleEvent(sim, start(sim) + P(sim)$predictionInterval, "lichenForecastYukon", "predictLichenPresence")
    },
    plot = {
      # ! ----- EDIT BELOW ----- ! #
      # do stuff for this event

      plotFun(sim) # example of a plotting function
      # schedule future event(s)

      # e.g.,
      #sim <- scheduleEvent(sim, time(sim) + P(sim)$.plotInterval, "lichenForecastYukon", "plot")

      # ! ----- STOP EDITING ----- ! #
    },
    save = {
      # ! ----- EDIT BELOW ----- ! #
      # do stuff for this event

      # e.g., call your custom functions/methods here
      # you can define your own methods below this `doEvent` function

      # schedule future event(s)

      # e.g.,
      # sim <- scheduleEvent(sim, time(sim) + P(sim)$.saveInterval, "lichenForecastYukon", "save")

      # ! ----- STOP EDITING ----- ! #
    },
    predictLichenPresence = {
      
      # Predict the presence/absence of caribou lichen for the provided input
      # csv. Predictions will be found in `testOutputs.csv`
      predictLichenPresence(sim, "mockInputs.csv", "testOutputs.csv") # Revisit naming here later
      
      # schedule future event(s)
      
      # e.g.,
      sim <- scheduleEvent(sim, time(sim) + P(sim)$predictionInterval, "lichenPredictYukon", "predictLichenPresence")
      
      # ! ----- STOP EDITING ----- ! #
    },
    warning(noEventWarning(sim))
  )
  return(invisible(sim))
}

### template initialization
### template initialization
Init <- function(sim) {
  # Run python script that creates lichen map using Google Earth Engine
  
  browser()
  # Get virtual env for Python
  installPyVirtualEnv("lichenPredictYukon")
  # Need to install packages for this virtual env?
  
  # Train ensemble classifier and generate a presence absence map over Yukon
  # Note need to save lichenData in googleDrive Folder! Put it there locally in interim  
  geoJSONPath <- file.path(paths(sim)$inputPath, "lichenData.geojson")
  outputPath <- file.path(paths(sim)$outputPath, "ensemble_prediction.tiff")
  cmd <- sprintf('import sys; sys.argv = ["%s", "%s", "%s"]', 
                 geoJSONPath, 
                 terra::res(sim$rasterToMatch)[[1]])
  py_run_string(cmd)
  GEEClassifierPath <- file.path(paths(sim)$modulePath[[1]], "lichenForecastYukon", "Python", "trainGEEClassifier.py")
  py_run_file(GEEClassifierPath)
  
  # NEED TO MERGE GEE MAP(ensemble_prediction.tiff) WITH LICHEN COVARIATES
  # Save static and dynamic covariates as a csv
  # NOTE THIS IS ONLY GIVING A DATA.TABLE FOR FORESTED PIXELS
  covariatesDT <- sim$lichenStaticCovariates[sim$lichenDynamicCovariates, on = "id", nomatch = 0]
  csvPath <- file.path(paths(sim)$inputPath, "lichenCovariates.csv")
  data.table::fwrite(covariatesDT, file = csvPath, row.names = FALSE)
  
  # Train classifier on forecastable covariates
  outputPath <- file.path(paths(sim)$outputPath, "bestModel.pkl")
  cmd <- sprintf('import sys; sys.argv = ["%s", "%s"]', csvPath, outputPath)
  py_run_string(cmd)
  trainClassifierPath <- file.path(paths(sim)$modulePath[[1]], "lichenForecastYukon", "Python", "trainForecastClassifier.py")
  py_run_file(trainClassifierPath)
  
  outputPath <- file.path(paths(sim)$outputPath, "lichenPrediction.csv")
  sim <- predictLichenPresence(sim, csvPath, outputPath)
  
  return(invisible(sim))
}
### template for save events
Save <- function(sim) {
  # ! ----- EDIT BELOW ----- ! #
  # do stuff for this event
  sim <- saveFiles(sim)
  
  # ! ----- STOP EDITING ----- ! #
  return(invisible(sim))
}

### template for plot events
plotFun <- function(sim, outputCSVName) {
  # ! ----- EDIT BELOW ----- ! #
  
  
  
  # ! ----- STOP EDITING ----- ! #
  return(invisible(sim))
}

predictLichenPresence <- function(sim, inputCSVName, outputCSVName) {
  
  ## Running the Python script to predict with a saved classifier
  cmd <- sprintf('import sys; sys.argv = ["%s", "%s"]', inputCSVName, outputCSVName)
  py_run_string(cmd)
  savedClassifierPath <- file.path(paths(sim)$modulePath[[1]], "lichenForecastYukon", "Python", "predictWithSavedClassifier.py")
  py_run_file(savedClassifierPath)
  
  return(invisible(sim))
}

.inputObjects <- function(sim) {
  # Any code written here will be run during the simInit for the purpose of creating
  # any objects required by this module and identified in the inputObjects element of defineModule.
  # This is useful if there is something required before simulation to produce the module
  # object dependencies, including such things as downloading default datasets, e.g.,
  # downloadData("LCC2005", modulePath(sim)).
  # Nothing should be created here that does not create a named object in inputObjects.
  # Any other initiation procedures should be put in "init" eventType of the doEvent function.
  # Note: the module developer can check if an object is 'suppliedElsewhere' to
  # selectively skip unnecessary steps because the user has provided those inputObjects in the
  # simInit call, or another module will supply or has supplied it. e.g.,
  # if (!suppliedElsewhere('defaultColor', sim)) {
  #   sim$map <- Cache(prepInputs, extractURL('map')) # download, extract, load file from url in sourceURL
  # }

  #cacheTags <- c(currentModule(sim), "function:.inputObjects") ## uncomment this if Cache is being used
  dPath <- asPath(getOption("reproducible.destinationPath", dataPath(sim)), 1)
  message(currentModule(sim), ": using dataPath '", dPath, "'.")

  # ! ----- EDIT BELOW ----- ! #

  # ! ----- STOP EDITING ----- ! #
  return(invisible(sim))
}

ggplotFn <- function(data, ...) {
  ggplot2::ggplot(data, ggplot2::aes(TheSample)) +
    ggplot2::geom_histogram(...)
}

