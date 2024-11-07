# Put any imports you need above these last ones as it changes the file path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))
from data_preprocessing import main as data_preprocessing_main

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

"""
Example of how to get cleaned and preprocessed data
data = data_preprocessing_main()

This data will be a dictionary where the key is the stock symbol (ex 'AAPL') and the value is another dictionary.
This nested dictionary has two keys 'stock_data' and 'news_data' where their values is a pandas data frame with column labels PC#
"""
def main():
    data = data_preprocessing_main()

    # Features
    X = data.drop(columns=['target'])

    # Values
    y = data['target']

    # Split data into training & test sets
    rand_state = 42 # Use a set random state for reproducible results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)  

    # Model, for classification
    model = RandomForestClassifier(n_estimators=100, random_state=rand_state) # Tune this with hyperparameters once testable
    model.fit(X_train, y_train) # Train model

    # Evaluate model accuracy using test data set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")

if __name__ == '__main__':
    main()