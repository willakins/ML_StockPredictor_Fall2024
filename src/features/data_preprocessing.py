import logging
import feature_reduction
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from data_collection import main as data_collection_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
Central location for running data preprocessing algorithms.
Currently, the only data preprocessing is dimensionality reduction, but more can be added here.

Current preprocessing algorithms:
    PCA
"""
def main():
    # Get collected and cleaned data
    data = data_collection_main()

    # n_components chosen semi-arbitrarily, probably should be tested and revisited
    # Creates an object to run all dimensionality reducing algorithms from
    feature_reducer = feature_reduction.FeatureReducer(data, n_components_stock = 5, n_components_news = 3)

    try:
        # Use PCA to reduce data's dimensions
        feature_reducer.use_pca()
        data = feature_reducer.get_stock_data()

        # Insert other algorithms here eventually

        return data
    except Exception as e:
        logger.error(f"Error in main data collection: {str(e)}")
        raise

if __name__ == '__main__':
    main()
    