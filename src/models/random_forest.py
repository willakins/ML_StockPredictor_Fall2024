# Put any imports you need above these last ones as it changes the file path
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


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
    # avg_sentiment, sentiment_std, news_count: Not implemented properly
    data.drop(columns=[
        'High', 'Low', 'Close', 'Volume', 'Dividends', 'news_count', 
        'Date', 'Stock Splits', 'Ticker',
        'avg_sentiment', 'sentiment_std', 'news_count'
    ], inplace=True)

    # Predicted Values
    y = data[['Target']]

    # Features
    # NOTE: I'm dropping Returns & Volume Change because we are using random forest CLASSIFIER
    # Target is a binary value, good for classification, but the others are numerical values
    # If we want to work with Returns & Volume_Change, use random forest REGRESSION
    X = data.drop(columns=['Returns', 'Volume_Change', 'Target'])
    logger.info(X.columns)
    # Split data into training & test sets
    rand_state = 42 # Use a set random state for reproducible results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)  

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Tune model hyperparameters by searching for best parameters
    # params = {
    #     'n_estimators': randint(50, 300),
    #     'max_depth': [None, 10, 20, 30, 40, 50],
    #     'min_samples_split': randint(2, 20),
    #     'min_samples_leaf': randint(1, 10),
    #     'max_features': ['sqrt', 'log2', None],
    #     'bootstrap': [True, False]
    # }

    # Search for best model parameters
    # model = RandomizedSearchCV(
    #     estimator=RandomForestClassifier(random_state=42),
    #     param_distributions=params,
    #     n_iter=50,
    #     cv=5,
    #     scoring='accuracy',
    #     n_jobs=1,
    #     random_state=rand_state,
    #     verbose=2
    # )

    # Create model using best parameters
    model = RandomForestClassifier(n_estimators=276,
                                   min_samples_split=2,
                                   min_samples_leaf=3,
                                   max_features=None,
                                   max_depth=50,
                                   bootstrap=False)
    
    logger.info('Training Model...')
    model.fit(X_train, y_train) # Train model

    # Evaluate model accuracy using test data set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy}")

    # logger.info(f"Best Model Accuracy: {model.best_score_}")
    # logger.info(f"Best Parameters: {model.best_params_}")
    # Best Model Accuracy: 0.5177
    # Best Parameters: 
    # {'bootstrap': False, 'max_depth': 50, 'max_features': None, 'min_samples_leaf': 3, 
    # 'min_samples_split': 2, 'n_estimators': 276}

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    logger.info(importance_df)

    logger.info('Creating plots...')

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Random Forest')
    plt.show()

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=50)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel('Permutation Feature Importance')
    plt.show()
    
    return model


if __name__ == '__main__':
    main()