# Put any imports you need above these last ones as it changes the file path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))
from data_preprocessing import main as data_preprocessing_main
"""
Example of how to get cleaned and preprocessed data
data = data_preprocessing_main()

This data will be a dictionary where the key is the stock symbol (ex 'AAPL') and the value is another dictionary.
This nested dictionary has two keys 'stock_data' and 'news_data' where their values is a pandas data frame with column labels PC#
"""
def main():
    data = data_preprocessing_main()

if __name__ == '__main__':
    main()