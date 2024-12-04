import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
import yaml
import logging

logging.basicConfig(level=logging.INFO)

class LSTMStockPredictor:
    def __init__(self, config_path='config/config.yaml', sequence_length=60):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['model']['lstm']
    
    def create_sequences(self, data, target):
        """Create sequences for LSTM input and corresponding targets.
        
        Args:
            data (np.array): Input features
            target (np.array): Target values
            
        Returns:
            tuple: (X, y) where X is the sequence data and y is the targets
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        # Store original prices for later use
        price_cols = [col for col in df.columns if col.startswith('PC')]
        tech_cols = [col for col in df.columns if not col.startswith('PC')]
        self.original_prices = df[price_cols].values
        
        # Scale prices to range (-1, 1)
        scaled_prices = self.price_scaler.fit_transform(df[price_cols])
        
        # Calculate percentage changes instead of returns
        price_changes = np.diff(scaled_prices, axis=0)
        
        # Scale technical indicators
        scaled_tech = self.feature_scaler.fit_transform(df[tech_cols])
        
        # Combine features
        combined_features = np.hstack([price_changes, scaled_tech[1:]])
        
        # Create sequences
        X, y = self.create_sequences(combined_features[:-1], price_changes[1:])
        
        # Split data without shuffling
        train_size = int(len(X) * 0.8)
        return (X[:train_size], X[train_size:],
                y[:train_size], y[train_size:])
    
    def build_model(self, input_shape):
        model = Sequential([
            # Simplified architecture
            LSTM(64, return_sequences=True, 
                 kernel_regularizer=l2(1e-5),
                 input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mse'
        )
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input sequences of shape (samples, sequence_length, features)
        Returns:
            numpy array of predicted prices
        """
        logging.info(f"Input X shape: {X.shape}")
        
        # Get predicted price changes
        changes_pred = self.model.predict(X)
        logging.info(f"Predicted changes shape: {changes_pred.shape}")
        
        # Get the correct window of original prices
        valid_end = len(self.original_prices) - self.sequence_length
        start_idx = valid_end - len(changes_pred)
        end_idx = valid_end
        price_window = self.original_prices[start_idx:end_idx]
        logging.info(f"Price window shape: {price_window.shape}")
        
        # Scale the price window
        scaled_price_window = self.price_scaler.transform(price_window)
        logging.info(f"Scaled price window shape: {scaled_price_window.shape}")
        
        # Expand changes_pred to match the number of price columns
        num_price_cols = scaled_price_window.shape[1]
        changes_expanded = np.repeat(changes_pred, num_price_cols, axis=1)
        logging.info(f"Expanded changes shape: {changes_expanded.shape}")
        
        # Calculate predicted scaled prices
        predicted_scaled = scaled_price_window + changes_expanded
        logging.info(f"Predicted scaled shape: {predicted_scaled.shape}")
        
        # Inverse transform to get actual prices
        predicted_prices = self.price_scaler.inverse_transform(predicted_scaled)
        logging.info(f"Final predictions shape: {predicted_prices[:, 0:1].shape}")
        
        # Store the valid indices for evaluation
        self.prediction_indices = slice(start_idx, end_idx)
        
        return predicted_prices[:, 0:1]  # Return only the first price column
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model predictions.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
        Returns:
            dict: Dictionary containing MAPE and RMSE metrics
        """
        # Use the same indices for true values as used in predictions
        y_true = y_true[self.prediction_indices, 0:1]
        
        # Verify shapes match
        logging.info(f"Evaluation shapes - True: {y_true.shape}, Pred: {y_pred.shape}")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
        
        # Remove any NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {
            'MAPE': mape,
            'RMSE': rmse
        }