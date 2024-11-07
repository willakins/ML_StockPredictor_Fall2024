import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureReducer:
    def __init__(self, stock_dict, n_components_stock, n_components_news):
        self.stock_data = stock_dict
        self.n_components_stock = n_components_stock
        self.n_components_news = n_components_news
        self.pca = PCA()    # Only Feature reducing algorithm used currently is PCA, but more could be added below 

    def use_pca(self):
        """
        Preprocess stock predictor data with PCA on both stock data and news data.
        
        Args:
            stock_dict (dict): Dictionary with stock symbols as keys and a nested dictionary containing
                                'stock_data' and 'news_data' (both are pd.DataFrame).
            n_components_stock: Number of principal components for stock data.
            n_components_news: Number of principal components for news data.
        
        Updates:
            self.stock_dict - dict: Dictionary with PCA-transformed 'stock_data' and 'news_data'.
        """
        for stock, data in self.stock_dict.items():
            # Get stock and news data
            stock_data = data['stock_data']
            news_data = data['news_data']

            # Fix missing values
            stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
            news_data = news_data.fillna(method='ffill').fillna(method='bfill')

            # Standardize stock data
            stock_scaler = StandardScaler()
            stock_data_scaled = stock_scaler.fit_transform(stock_data)
            
            # Standardize news data
            news_scaler = StandardScaler()
            news_data_scaled = news_scaler.fit_transform(news_data)

            # Apply PCA on stock data
            pca_stock = PCA(n_components = self.n_components_stock)
            stock_data_pca = pca_stock.fit_transform(stock_data_scaled)
            stock_data_pca = pd.DataFrame(stock_data_pca, columns=[f'PC{i+1}_stock' for i in range(self.n_components_stock)])

            # Apply PCA on news data
            pca_news = PCA(n_components = self.n_components_news)
            news_data_pca = pca_news.fit_transform(news_data_scaled)
            news_data_pca = pd.DataFrame(news_data_pca, columns=[f'PC{i+1}_news' for i in range(self.n_components_news)])

            # Update dictionary with PCA-transformed data
            self.stock_dict[stock]['stock_data'] = stock_data_pca
            self.stock_dict[stock]['news_data'] = news_data_pca

    """
    Getter for stock_data once its dimensions have been reduced

    Returns:
        self.stock_data - dict: Dictionary with dimension reduced stock data 
    """
    def get_stock_data(self):
        return self.stock_data
