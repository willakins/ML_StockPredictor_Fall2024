import logging

import yaml
import feature_reduction
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main class for preprocessing stock market and news data."""
    def __init__(self, n_components_stock, n_components_news, config_path: str = 'config/config.yaml', 
                 data_path: str = 'data/processed', data_saving_path: str = 'data/preprocessed'):
        """
        Initialize the data collector with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_path = data_path
        self.data_saving_path = data_saving_path
    
        self.data = {}
        self.n_components_stock = n_components_stock
        self.n_components_news = n_components_news
        # Access data configuration
        self.symbols = self.config['data'].get('ticker_list', [])  # Changed from 'symbols'
        if not self.symbols:  # If ticker_list is empty, try to get from stock section
            self.symbols = self.config['data'].get('stock', {}).get('ticker_list', [])
    
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            raise
    
    """
    Gets the cleaned data from the csv and converts it into a pandas data frame
    """
    def collect_data(self):
        for symbol in self.symbols:
            stock_data = pd.read_csv(self.data_path + f'/{symbol}_stock_processed.csv')
            news_data = pd.read_csv(self.data_path + f'/{symbol}_news_processed.csv')
            self.data[symbol] = {
                'stock_data': stock_data,
                'news_data': news_data
            }

    """
    Where feature reduction algorithms are run
    """
    def reduce_features(self):
        feature_reducer = feature_reduction.FeatureReducer(self.data, self.n_components_stock, self.n_components_news)
        try:
            # Use PCA to reduce data's dimensions
            feature_reducer.use_pca()
            self.data = feature_reducer.get_stock_data()

            for symbol in self.symbols:
                stock_data = self.data[symbol]['stock_data']
                news_data = self.data[symbol]['news_data']
                # Save processed data
                if not stock_data.empty:
                    stock_data.to_csv(self.data_saving_path + f"/{symbol}_stock_preprocessed.csv")
                if not news_data.empty:
                    news_data.to_csv(self.data_saving_path + f"/{symbol}_news_preprocessed.csv")
        except Exception as e:
            logger.error(f"Error in reducing features: {str(e)}")
            raise

    def get_data(self):
        return self.data

def main():
    """Main function to run data preprocessing."""
    try:
        # n_components chosen semi-arbitrarily, probably should be tested and revisited
        preprocessor = DataProcessor(n_components_stock = 5, n_components_news = 1)
        preprocessor.collect_data()
        preprocessor.reduce_features()
        # Insert other algorithms here eventually

        return preprocessor.get_data()
    except Exception as e:
        logger.error(f"Error in main data preprocessing: {str(e)}")
        raise

if __name__ == '__main__':
    main()
    