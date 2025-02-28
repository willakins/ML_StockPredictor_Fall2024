# Stock Market Trend Prediction AI Model

## Introduction

The stock market is a complex adaptive system that reacts to various global events, making stock market trend prediction a highly challenging yet essential task for investors. This project tackles the challenge by combining two types of data: historical stock price data and news sentiment data. By using a hybrid approach, this AI model attempts to predict short-term stock price movements (1-5 days) for S&P 500 companies.

## Problem Definition

Accurate stock market prediction is crucial for optimizing investment strategies, managing financial risks, and understanding market dynamics. The task is to combine numerical time-series data with textual sentiment information to predict short-term stock price movements.

## Data Collection and Preprocessing

The project uses two key data sources:
- **Yahoo Finance API**: Provides historical stock price data for S&P 500 companies.
- **Alpha Vantage News API**: Supplies real-time and historical financial news articles.

To preprocess the data:
1. **Stock Data**: Technical indicators were computed using Yahoo Finance API, and custom functions were developed to calculate advanced indicators.
2. **News Data**: Text from financial news articles was cleaned, tokenized, and analyzed for sentiment, polarity, and subjectivity, which were then used as features in the model.

Principal Component Analysis (PCA) was applied to reduce the dimensionality of the data, making it easier to debug and visualize.

## ML Models

Three distinct machine learning models were applied to this problem:

1. **LSTM (Long Short-Term Memory)**: Ideal for sequential data and capturing temporal dependencies in stock price movements.
2. **Random Forest Classifier**: An ensemble learning model that combines multiple decision trees, robust to noisy data and overfitting.
3. **Support Vector Machine (SVM)**: A powerful model for classification tasks, particularly effective when handling high-dimensional data and complex decision boundaries.

### Model Overview:

- **LSTM**: Perfect for capturing the temporal nature of stock prices. It achieved the highest accuracy of 96%.
- **Random Forest Classifier**: Performed poorly, with an accuracy of ~50%. Feature selection (via Permutation Feature Importance) showed that certain features were either overfitting or ineffective.
- **SVM**: Achieved an average accuracy of 85%. It was able to handle high-dimensional data well but struggled with temporal dependencies inherent in stock prices.

## Quantitative Metrics

- **MAPE (Mean Absolute Percentage Error)**: Measures the average percentage difference between predicted and actual values.
- **RMSE (Root Mean Square Error)**: The square root of the average squared differences between predicted and actual values.
- **Accuracy**: Proportion of correct predictions (both upward and downward movements).
- **F1 Score**: Balances precision and recall, particularly useful in imbalanced datasets.

## Results and Discussion

### Long Short-Term Memory (LSTM)
- **Accuracy**: 96%
- LSTM performed the best, capturing long-term trends and short-term fluctuations. However, it exhibited high error variance, especially for more volatile stocks.

### Random Forest Classifier
- **Accuracy**: ~50%
- Random Forest performed poorly due to inefficient feature selection. Feature importance analysis suggested that removing some features (like SMA5 and SMA20) did not improve performance.

### Support Vector Machine (SVM)
- **Accuracy**: 85%
- SVM showed decent performance, though it had higher RMSE values (0.35-0.42). It struggled to capture temporal dependencies like LSTM but performed reasonably well with high-dimensional data.

## Key Findings

- **LSTM**: Best for capturing temporal dependencies and trends. Works well with stable stocks like AAPL.
- **Random Forest**: Struggles with stock market data due to feature selection issues, despite being robust to noisy data.
- **SVM**: Provides good balance between precision and recall, performing well with well-structured data but lacks the ability to model sequential data effectively.

## Next Steps

The next phase involves integrating the strengths of all three models into a unified framework. By combining time-series analysis, sentiment interpretation, and classification, this hybrid model could provide a more accurate and reliable prediction system. Further experiments will explore improvements in feature engineering and model design.

## References

1. Li, J., Li, G., Liu, M., Zhu, X., & Wei, L. (2022). A novel text-based framework for forecasting agricultural futures using massive online news headlines. *International Journal of Forecasting*, 38(1), 35-50. [DOI: 10.1016/j.ijforecast.2020.02.002](https://doi.org/10.1016/j.ijforecast.2020.02.002)
2. Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques. *Expert Systems with Applications*, 42(1), 259-268. [DOI: 10.1016/j.eswa.2014.07.040](https://doi.org/10.1016/j.eswa.2014.07.040)
3. Feng, F., Chen, H., He, X., Ding, J., Sun, M., & Chua, T.-S. (2019). Enhancing Stock Movement Prediction with Adversarial Training. *arXiv.org*. [Link](https://arxiv.org/abs/1810.09936v2)

## Installation

```bash
# Clone the repository
git clone https://github.com/willakins/ML_StockPredictor_Fall2024
cd stock_market_prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For development installations:
pip install -e ".[dev]"

## Project Structure
stock_market_prediction/
│
├── data/                    # Data directory
│   ├── raw/                # Raw data downloads
│   ├── processed/          # Cleaned and processed data
│   └── external/           # External data sources
│
├── src/                    # Source code
│   ├── data/              # Data collection and processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── visualization/     # Data visualization
│
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── config/               # Configuration files
├── requirements.txt      # Project dependencies
└── setup.py             # Package installation


## Keys
Alpha Vantage API key: 5O7VTPF7G6OFH4L4
