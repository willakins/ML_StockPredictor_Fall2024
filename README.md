# Stock Market Prediction

A machine learning project for predicting stock market trends using historical price data and sentiment analysis.

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