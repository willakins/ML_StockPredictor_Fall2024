import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yaml
import logging

logging.basicConfig(level=logging.INFO)

class SVMStockPredictor:
    def __init__(self, config_path='config/config.yaml'):
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        # Tune hyperparameters?
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['models']['svm']
    
    def prepare_data(self, df):
        """
        Prepare data for SVM model: scale features and create binary targets.
        """
        # Identify price-related and technical indicator columns
        price_cols = [col for col in df.columns if col.startswith('PC')]
        tech_cols = [col for col in df.columns if not col.startswith('PC')]
        
        # Store original prices for evaluation later
        self.original_prices = df[price_cols].values

        # Scale prices to range (-1, 1)
        scaled_prices = self.price_scaler.fit_transform(df[price_cols])
        
        # Calculate percentage changes to use as the target
        price_changes = np.diff(scaled_prices, axis=0)
        binary_target = (price_changes[:, 0] > 0).astype(int)  # 1 if price increases, else 0
        
        # Scale technical indicators
        scaled_tech = self.feature_scaler.fit_transform(df[tech_cols])
        
        # Combine features
        combined_features = np.hstack([price_changes, scaled_tech[1:]])
        
        # Ensure target length matches feature length
        binary_target = binary_target[:len(combined_features)]
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, binary_target, test_size=0.2, random_state=42, stratify=binary_target
        )
    
        return X_train, X_test, y_train, y_test

    
    def train(self, X_train, y_train):
        """
        Train the SVM model.
        """
        logging.info("Training SVM model...")
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Make predictions with the trained SVM model.
        """
        logging.info("Predicting with SVM model...")
        return self.model.predict(X_test)
    
    def predict_probabilities(self, X_test):
        """
        Predict probabilities for the classes.
        """
        logging.info("Predicting probabilities...")
        return self.model.predict_proba(X_test)[:, 1]  # Probability for class 1
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance with classification metrics and regression-like metrics (MAPE, RMSE).
        """
        # Classification Metrics
        logging.info("Evaluating model...")
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
        
        # Regression-like Metrics (MAPE and RMSE)
        # mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Avoid division by zero
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # logging.info(f"MAPE: {mape:.2f}%")
        logging.info(f"RMSE: {rmse:.2f}")
        
        return {
            'Accuracy': accuracy,
            'Classification Report': report,
            # 'MAPE': mape,
            'RMSE': rmse
        }

