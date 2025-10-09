import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from xgboost import XGBClassifier
import sys

def main():
  if len(sys.argv) != 2:
      print("Incorrect number of sys inputs to `trainForecastClassifier.py`")
      sys.exit(1)
      
  # Reading in the data
  df = pd.read_csv(sys.argv[0])
  df = df.dropna(subset=[c for c in df.columns if c != "pixelGroup"])
  
  # Splitting training and test set
  train_set, test_set = train_test_split(
      df, 
      test_size=0.2, 
      random_state=42,
      stratify=df["lichenPresence"]
  )
  
  # Separate features and target
  X_train = train_set.drop(columns=["lichenPresence", "pixelGroup"])
  y_train = train_set["lichenPresence"].astype(int)
  X_test = test_set.drop(columns=["lichenPresence", "pixelGroup"])
  y_test = test_set["lichenPresence"].astype(int)
  
  # Finding bias of the dataset
  frequencies = y_train.value_counts()
  scale_pos_weight = frequencies[0] / frequencies[1]
  
  # One-hot encode categorical variables
  X_train = pd.get_dummies(X_train, drop_first=True)
  X_test = pd.get_dummies(X_test, drop_first=True)
  X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
  
  # Define the grid of hyperparameters (lists of values)
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [3, 6, 10],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.8, 1.0],
      'colsample_bytree': [0.8, 1.0]
  }
  
  # Set up RandomizedSearchCV to sample over the grid
  random_search = RandomizedSearchCV(
      XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42),
      param_distributions=param_grid,
      n_iter=20,
      cv=5,
      n_jobs=20,
      verbose=2,
      random_state=42
  )
  
  # Fit random search
  random_search.fit(X_train, y_train)
  
  # Best estimator and parameters
  print("Best parameters:", random_search.best_params_)
  clf = random_search.best_estimator_
  
  # Predict and evaluate
  y_pred = clf.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred))
  
  # Save best model
  joblib.dump(clf, sys.argv[1])
  print(f"Best model saved to `{sys.argv[1]}`")
  
  # Best Params RF Best parameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
  # Accuracy ~0.7 on both classes
  
  # Best Params XGBoost: Best parameters: {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.2, 'colsample_bytree': 1.0}
  # Accuracy ~0.7 on both
  
if __name__=="__main__":
  main()
