"""
Data collection module for stock market prediction project.
Handles fetching and initial processing of stock and news data.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Union
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
import nltk
import ssl
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Main class for collecting stock market and news data."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the data collector with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Access data configuration with error handling
        try:
            self.symbols = self.config.get('data', {}).get('ticker_list', [])
            self.start_date = self.config.get('data', {}).get('start_date')
            self.end_date = self.config.get('data', {}).get('end_date')
            
            # Get API configuration
            self.api_key = self.config.get('api', {}).get('alpha_vantage_key')
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not found in config")
            
            # Get feature configuration
            features = self.config.get('features', {})
            self.technical_indicators = features.get('technical_indicators', [])
            self.sentiment_enabled = features.get('sentiment_analysis', {}).get('enabled', False)

            # Initialize paths
            self.raw_data_path = Path('data/raw')
            self.processed_data_path = Path('data/processed')
            
            # Create directories if they don't exist
            self._setup_directories()
            
            # Setup NLP if sentiment analysis is enabled
            if self.sentiment_enabled:
                self._setup_nlp()
            
        except Exception as e:
            logger.error(f"Error initializing DataCollector with config: {str(e)}")
            raise

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            raise

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def _setup_nlp(self):
        """Setup NLTK and initialize NLP tools."""
        try:
            for package in ['punkt', 'stopwords', 'wordnet']:
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context

                try:
                    nltk.data.find(f'tokenizers/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords)
        except Exception as e:
            logger.error(f"Error setting up NLP tools: {str(e)}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on config."""
        for indicator in self.technical_indicators:
            try:
                if indicator == 'SMA':
                    df['SMA20'] = df['Close'].rolling(window=20).mean()
                    df['SMA50'] = df['Close'].rolling(window=50).mean()
                elif indicator == 'RSI':
                    df['RSI'] = self._calculate_rsi(df['Close'])
                elif indicator == 'MACD':
                    df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {str(e)}")
        return df

    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical stock data for a single symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        logger.info(f"Fetching stock data for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)
            
            # Add basic price features
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Calculate configured technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Save raw data
            df.to_csv(self.raw_data_path / f"{symbol}_stock_data.csv")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_news_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch news data for a single symbol from Alpha Vantage.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: News data
        """
        logger.info(f"Fetching news data for {symbol}")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            if 'feed' in data:
                df = pd.DataFrame(data['feed'])
                # Save raw data
                df.to_csv(self.raw_data_path / f"{symbol}_news_data.csv")
                return df
            elif 'Note' in data:  # API limit reached
                logger.warning(f"API limit reached: {data['Note']}")
                time.sleep(60)  # Wait for a minute before retrying
                return self.fetch_news_data(symbol)  # Retry
            
            return pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _calculate_macd(prices: pd.Series, 
                       short_window: int = 12,
                       long_window: int = 26,
                       signal_window: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def collect_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect both stock and news data for all symbols.
        
        Returns:
            Dict containing stock and news data for each symbol
        """
        all_data = {}
        
        for symbol in self.symbols:
            logger.info(f"Processing data for {symbol}")
            
            # Collect stock data
            stock_data = self.fetch_stock_data(symbol)
            
            # Collect news data if sentiment analysis is enabled
            news_data = pd.DataFrame()
            if self.sentiment_enabled:
                news_data = self.fetch_news_data(symbol)
                
                if not news_data.empty:
                    # Process news data
                    news_data['cleaned_text'] = news_data['title'].apply(self._clean_text)
                    news_data['sentiment'] = news_data['cleaned_text'].apply(self._calculate_sentiment)
                    news_data['polarity'] = news_data['sentiment'].apply(lambda x: x['polarity'])
                    news_data['subjectivity'] = news_data['sentiment'].apply(lambda x: x['subjectivity'])
            
            all_data[symbol] = {
                'stock_data': stock_data,
                'news_data': news_data
            }
            
            # Save processed data
            if not stock_data.empty:
                stock_data.to_csv(self.processed_data_path / f"{symbol}_stock_processed.csv")
            if not news_data.empty:
                news_data.to_csv(self.processed_data_path / f"{symbol}_news_processed.csv")
        
        return all_data

    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        try:
            # Convert to lowercase and tokenize
            tokens = word_tokenize(str(text).lower())
            
            # Remove stopwords and lemmatize
            cleaned_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalnum() and token not in self.stop_words
            ]
            
            return ' '.join(cleaned_tokens)
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return str(text)

    def _calculate_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores for text."""
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}

def main():
    """Main function to run data collection."""
    try:
        collector = DataCollector()
        data = collector.collect_all_data()
        logger.info("Data collection completed successfully")
        return data
    except Exception as e:
        logger.error(f"Error in main data collection: {str(e)}")
        raise

if __name__ == "__main__":
    main()