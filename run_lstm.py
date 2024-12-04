import pandas as pd
import numpy as np
from src.models.LSTM import LSTMStockPredictor
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(symbol):
    try:
        stock_file = f'data/preprocessed/{symbol}_stock_preprocessed.csv'
        if not os.path.exists(stock_file):
            raise FileNotFoundError(f"Could not find preprocessed data for {symbol}")
        df = pd.read_csv(stock_file)
        return df
    except Exception as e:
        logging.error(f"Error loading data for {symbol}: {str(e)}")
        raise

def main():
    try:
        symbols = ['AAPL', 'GOOG', 'MSFT']
        results = []
        predictions_dict = {}
        
        for symbol in symbols:
            logging.info(f"Processing {symbol}...")
            df = load_data(symbol)
            predictor = LSTMStockPredictor(sequence_length=60)
            logging.info("Preparing data...")
            X_train, X_val, y_train, y_val = predictor.prepare_data(df)
            logging.info("Training model...")
            history = predictor.train(X_train, y_train, X_val, y_val)
            logging.info("Making predictions...")
            predictions = predictor.predict(X_val)
            
            # Store predictions
            predictions_dict[symbol] = predictions
            
            metrics = predictor.evaluate(predictor.original_prices, predictions)
            results.append({
                'Symbol': symbol,
                'MAPE': metrics['MAPE'],
                'RMSE': metrics['RMSE'],
                'Accuracy': metrics['Accuracy'],
                'F1 Score': metrics['F1 Score']
            })
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('data/lstm_results.csv', index=False)
        
        # Save predictions for each symbol
        for symbol, preds in predictions_dict.items():
            pd.DataFrame(preds, columns=['Predicted_Price']).to_csv(
                f'data/lstm_predictions_{symbol}.csv', index=False)
            
        logging.info("Results saved to data/lstm_results.csv")
        logging.info("Predictions saved as lstm_predictions_[SYMBOL].csv")
        
        logging.info("Final Results:")
        for result in results:
            logging.info(f"{result['Symbol']} - MAPE: {result['MAPE']:.2f}%, RMSE: {result['RMSE']:.2f}, "
                        f"Accuracy: {result['Accuracy']:.4f}, F1 Score: {result['F1 Score']:.4f}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()