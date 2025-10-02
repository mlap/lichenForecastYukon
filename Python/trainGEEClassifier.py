#!/usr/bin/env python
# coding: utf-8
import ee
import geemap
import json
import random
import itertools
from ee.ee_exception import EEException
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def main():
  # Initialize the Earth Engine module.
  ee.Authenticate()
  ee.Initialize()
  
  # Get Yukon geometry from GADM
  gadm = ee.FeatureCollection("FAO/GAUL/2015/level1")
  yukon = gadm.filter(ee.Filter.eq('ADM1_NAME', 'Yukon')).first()
  geometry = yukon.geometry()
  
  
  # Getting lichen presence data from a local GeoJSON file
  geojson_path = "./caribouLichen.geojson"
  
  # Convert GeoJSON to Earth Engine FeatureCollection
  fc = geemap.geojson_to_ee(geojson_path)
  
  fc_random = fc.randomColumn('random', seed=42)
  
  # Making train and test datasets
  positive_samples = fc_random.filter(ee.Filter.eq('presenceCaribouLichen', 1))
  negative_samples = fc_random.filter(ee.Filter.eq('presenceCaribouLichen', 0))
  
  
  # Randomly sample points
  train_positives = positive_samples.filter(ee.Filter.lt('random', 0.8))
  train_negatives = negative_samples.filter(ee.Filter.lt('random', 0.8))
  
  # Trying resampling
  num_positives = train_positives.size()
  num_negatives = train_negatives.size()
  
  training_ = train_positives.merge(train_negatives)
  
  test_positives = positive_samples.filter(ee.Filter.gte('random', 0.8))
  test_negatives = negative_samples.filter(ee.Filter.gte('random', 0.8))
  test_ = test_positives.merge(test_negatives)
  
  # Pick a year for classification
  year = 2024
  startDate = ee.Date.fromYMD(year, 1, 1)
  endDate = startDate.advance(1, 'year')
  
  # Load the Satellite Embeddings image collection and filter by date and region
  embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  embeddingsFiltered = embeddings.filter(ee.Filter.date(startDate, endDate)).filter(ee.Filter.bounds(geometry))
  embeddingsImage = embeddingsFiltered.mosaic()
  
  # Sample the embeddings image at the training points
  training = embeddingsImage.sampleRegions(
      collection=training_,
      properties=['presenceCaribouLichen'],
      scale=10
  )
  
  # List of classifiers and their parameter grids
  # to sample over for hyperparameter tuning
  model_grid = {
      'smileRandomForest': {
          'numberOfTrees': [10, 50, 100],
          'variablesPerSplit': [None, 5, 10, 20],
          'minLeafPopulation': [1, 5, 10],
          'bagFraction': [0.5, 0.7, 0.9],
          'maxNodes': [None, 10, 20]
      },
      'smileGradientTreeBoost': {
          'numberOfTrees': [10, 50, 100],
          'shrinkage': [0.005, 0.01, 0.1, 0.3],
          'samplingRate': [0.5, 0.7, 0.9],
          'maxNodes': [None, 10, 20],
          'loss': ['LeastAbsoluteDeviation', 'Huber'],
      },
      'smileKNN': {
          'k': [1, 3, 5, 8],
          'searchMethod': ['AUTO', 'KD_TREE', 'COVER_TREE'],
          'metric': ['EUCLIDEAN', 'MANHATTAN', 'MAHALANOBIS'],
      },
      'smileNaiveBayes': {
          'lambda': [0.000001, 0.0001, 0.001, 0.01, 0.1],
      },
      'smileCart': {
          'maxNodes': [None, 10, 20],
          'minLeafPopulation': [1, 5, 10],
      }    
  }
  
  best_acc = 0
  best_model = None
  best_params = None
  
  results = []
  num_samples_ = 10
  
  # Loop over models and parameter combinations
  for model_name, param_grid in model_grid.items():
      param_keys = list(param_grid.keys())
      param_values = list(param_grid.values())
      all_combos = []
      for combo in itertools.product(*param_values):
          param_dict = dict(zip(param_keys, combo))
          all_combos.append(param_dict)
      num_samples = min(num_samples_, len(all_combos))
      sampled_combos = random.sample(all_combos, num_samples)
      for params in sampled_combos:
          print(f"Now performing hyperparameter tuning for {model_name} with params: {params}")
          try:
              classifier = getattr(ee.Classifier, model_name)(**params).train(
                  features=training,
                  classProperty='presenceCaribouLichen',
                  inputProperties=embeddingsImage.bandNames()
              )
  
              classified = embeddingsImage.classify(classifier)
  
              test_samples = classified.sampleRegions(
                  collection=test_,
                  properties=['presenceCaribouLichen'],
                  scale=10
              )
  
              conf_matrix = test_samples.errorMatrix('presenceCaribouLichen', 'classification')
              acc = conf_matrix.accuracy().getInfo()
  
          except EEException as e:
              print(f"Error training {model_name} with params {params}: {e}")
              continue
  
  
          print(f'{acc:.3f} Accuracy for {model_name} with params {params}')
  
          results.append({
              'model': model_name,
              'params': params,
              'accuracy': acc
          })
  
          if acc > best_acc:
              best_acc = acc
              best_model = model_name
              best_params = params
  
  # Sort results by accuracy in descending order
  top10 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]
  
  # Retrain top 10 models and get their probability of lichen presence
  proba_images = []
  for res in top10:
      model_name = res['model']
      params = res['params']
      classifier = getattr(ee.Classifier, model_name)(**params).train(
          features=training,
          classProperty='presenceCaribouLichen',
          inputProperties=embeddingsImage.bandNames()
      )
      # Get probability band (if supported)
      classified = embeddingsImage.classify(classifier, outputName='probability')
      proba_images.append(classified.select('probability'))
  
  # Average probabilities and create an ensemble prediction
  # acoording to a threshold
  ensemble_proba = ee.ImageCollection(proba_images).mean()
  ensemble_prediction = ensemble_proba.gt(0.3) # Might want to change this threshold
  
  # Sample ensemble prediction at test points
  ensemble_test_samples = ensemble_prediction.rename('classification').sampleRegions(
      collection=test_,
      properties=['presenceCaribouLichen'],
      scale=10
  )
  
  # Compute confusion matrix
  ensemble_conf_matrix = ensemble_test_samples.errorMatrix('presenceCaribouLichen', 'classification')
  
  # Print accuracy and per-class metrics
  print('Ensemble Accuracy:', ensemble_conf_matrix.accuracy().getInfo())
  print('Ensemble Kappa:', ensemble_conf_matrix.kappa().getInfo())
  pa = ensemble_conf_matrix.producersAccuracy()
  print('Negative Class Accuracy:', pa.get([0, 0]).getInfo())
  print('Positive Class Accuracy:', pa.get([1, 0]).getInfo())
  
  # Throughout the rest of the code block, I export the raster to GEE and then
  # download it locally
  
  # Export raster to Google Earth Engine
  export_task = ee.batch.Export.image.toDrive(
      image=ensemble_prediction,
      description='ensemble_prediction_export',
      folder='EarthEngineExports',  # Change as needed
      fileNamePrefix='ensemble_prediction',
      region=geometry.bounds().getInfo()['coordinates'],
      scale=240,  # or 240 if you want 240m pixels
      crs='EPSG:3579',  # or match your desired projection
      maxPixels=1e13,
      fileFormat='GeoTIFF'
  )
  export_task.start()
  
  # Authenticate and create the PyDrive client
  gauth = GoogleAuth()
  gauth.LocalWebserverAuth()  # Follow the link in the output to authenticate
  drive = GoogleDrive(gauth)
  
  # List files in the EarthEngineExports folder
  folder_name = 'EarthEngineExports'
  file_prefix = 'ensemble_prediction'
  
  # Find the folder ID for EarthEngineExports
  file_list = drive.ListFile({'q': "mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
  folder_id = None
  for folder in file_list:
      if folder['title'] == folder_name:
          folder_id = folder['id']
          break
  
  if folder_id is None:
      raise Exception(f"Folder '{folder_name}' not found in your Google Drive.")
  
  # List files in the folder and download the one with the prefix
  file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
  for file in file_list:
      if file['title'].startswith(file_prefix):
          print(f"Downloading {file['title']} ...")
          file.GetContentFile(file['title'])
          print(f"Downloaded {file['title']} to {os.getcwd()}")

if __name__=="__main__":
  main()

