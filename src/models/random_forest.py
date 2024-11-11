# Put any imports you need above these last ones as it changes the file path
import sys
import os
import logging
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():

    # I'm not sure what to do with the preprocessed data (PC1-5??), so I'm just going to work with the processed data
    data = pd.read_csv('data/processed/combined_processed_data.csv')

    # I'm going to drop a few columns (with justification)

    # Some of these columns require insight on how the stock will perform throughout the whole day
    # We can use these as target (predicted) values, but not as data (predictor) values.
    # So, unless a predicted column is a good indicator of stock performance, I'm going to remove it
    # The following columns are dropped for this reason:

    # High, Low, Close, Volume, Dividends, news_count

    # Other dropped columns specified here:
    # Date: Random Forest doesn't handle sequential data
    # Stock Splits: MAYBE put this one back, idk i dont rly get this one. Not sure if predicted or predictor
    # Ticker: Don't need that
    data.drop(columns=[
        'High', 'Low', 'Close', 'Volume', 'Dividends', 'news_count', 
        'Date', 'Stock Splits', 'Ticker'
    ], inplace=True)

    # Predicted Values
    y = data[['Target']]

    # Features
    # NOTE: I'm dropping Returns & Volume Change because we are using random forest CLASSIFIER
    # Target is a binary value, good for classification, but the others are numerical values
    X = data.drop(columns=['Returns', 'Volume_Change', 'Target'])

    # Split data into training & test sets
    rand_state = 42 # Use a set random state for reproducible results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)  

    # Model (NOT classification!)
    model = RandomForestClassifier(n_estimators=100, random_state=rand_state) # Tune this with hyperparameters once testable
    model.fit(X_train, y_train) # Train model

    # Evaluate model accuracy using test data set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy}")

    return model

if __name__ == '__main__':
    main()