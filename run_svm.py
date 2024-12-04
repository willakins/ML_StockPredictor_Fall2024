import pandas as pd
import numpy as np
from src.models.support_vector_machine import SVMStockPredictor
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(symbol):
    """Load and combine stock and preprocessed data for a given symbol."""
    try:
        # Construct file paths
        stock_file = f'data/preprocessed/{symbol}_stock_preprocessed.csv'
        
        if not os.path.exists(stock_file):
            raise FileNotFoundError(f"Could not find preprocessed data for {symbol}")
        
        # Load the data
        df = pd.read_csv(stock_file)
        
        return df
    except Exception as e:
        logging.error(f"Error loading data for {symbol}: {str(e)}")
        raise

def main():
    try:
        # List of symbols to process
        symbols = ['AAPL', 'GOOG', 'MSFT']  # Add or modify symbols as needed
        
        for symbol in symbols:
            logging.info(f"Processing {symbol}...")
            
            # Load data
            df = load_data(symbol)
            
            # Initialize predictor
            predictor = SVMStockPredictor()
            
            # Prepare data
            logging.info("Preparing data...")
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            
            # Train model
            logging.info("Training model...")
            predictor.train(X_train, y_train)
            
            # Make predictions
            logging.info("Making predictions...")
            y_pred = predictor.predict(X_test)
            
            # Evaluate
            metrics = predictor.evaluate(y_test, y_pred)
            logging.info(f"Results for {symbol}:")
            logging.info(f"Accuracy: {metrics['Accuracy']:.2f}")
            # logging.info(f"MAPE: {metrics['MAPE']:.2f}%")
            logging.info(f"RMSE: {metrics['RMSE']:.2f}")
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Date': df.index[-len(y_test):]  # Assume last indices correspond to test set
            })
            
            # Save results
            output_dir = 'results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'{output_dir}/{symbol}_predictions_{timestamp}.csv'
            results_df.to_csv(results_file, index=False)
            logging.info(f"Results saved to {results_file}")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
