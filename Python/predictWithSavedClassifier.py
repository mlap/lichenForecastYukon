import pandas as pd
import joblib
import sys

# Usage: python predictWithSavedModel.py input.csv output.csv
def main():
  if len(sys.argv) != 3:
      print("Incorrect number of sys inputs to `predictWithSavedClassifier.py`")
      sys.exit(1)
  
  input_csv = sys.argv[0]
  output_csv = sys.argv[1]
  
  # Load the saved model
  clf = joblib.load(sys.argv[2])
  
  # Read input data
  df = pd.read_csv(input_csv)
  df_out = df.copy()
  df = df.dropna(subset=[c for c in df.columns if c != "pixelGroup"])
  
  # One-hot encode categorical variables
  inputs = pd.get_dummies(df, drop_first=True)
  
  # Align columns with training data
  model_features = clf.feature_names_in_
  inputs = inputs.reindex(columns=model_features, fill_value=0)
  
  # Make predictions
  preds = clf.predict(inputs)
  probs = clf.predict_proba(inputs)[:, 1]  # Probability of class 1
  
  # Save predictions
  df["predictedClass"] = preds
  df["positiveClassProbability"] = probs
  df_out = df_out.join(df[["predictedClass", "positiveClassProbability"]], how="left")
  df_out[["predictedClass", "positiveClassProbability"]].to_csv(output_csv, index=False)
  print(f"Predictions saved to {output_csv}")
  
if __name__=="__main__":
  main()
