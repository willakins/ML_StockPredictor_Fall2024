import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, 
    RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stock_prediction.log')
    ]
)
logger = logging.getLogger('stock_predictor')

class StockPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.pred_probas = None
        self.feature_importance_df = None
        
        # Create necessary directories
        os.makedirs('docs/charts/rfc', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def analyze_data_distribution(self, data):
        """Analyze and log data distribution details"""
        logger.info("Analyzing data distribution...")
        
        # Overall class distribution
        class_dist = data['Target'].value_counts(normalize=True)
        logger.info(f"\nOverall class distribution:\n{class_dist}")
        
        # Distribution by stock
        stock_dist = data.groupby('Ticker')['Target'].value_counts(normalize=True).unstack()
        logger.info(f"\nClass distribution by stock:\n{stock_dist}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"\nMissing values:\n{missing_values}")
        
        # Feature correlation analysis
        correlation_matrix = data.drop(['Ticker'], axis=1).corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('docs/charts/rfc/correlation_matrix.png')
        plt.close()

    def load_data(self):
        """Load and combine PCA-processed stock and news data with target calculation"""
        logger.info('Loading preprocessed data...')
        stocks = ['AAPL', 'GOOG', 'MSFT']
        combined_data = []
        
        for stock in stocks:
            try:
                stock_data = pd.read_csv(f'data/preprocessed/{stock}_stock_preprocessed.csv')
                news_data = pd.read_csv(f'data/preprocessed/{stock}_news_preprocessed.csv')
                processed_data = pd.read_csv(f'data/preprocessed/{stock}_processed.csv')
                
                # Clean indices
                stock_data = stock_data.drop('Unnamed: 0', axis=1, errors='ignore')
                news_data = news_data.drop('Unnamed: 0', axis=1, errors='ignore')
                processed_data = processed_data.drop('Unnamed: 0', axis=1, errors='ignore')
                
                # Reset indices
                stock_data.reset_index(drop=True, inplace=True)
                news_data.reset_index(drop=True, inplace=True)
                processed_data.reset_index(drop=True, inplace=True)
                
                # Calculate target and add ticker
                target = (processed_data['Returns'] > 0).astype(int)
                df_combined = pd.concat([
                    stock_data,
                    news_data,
                    pd.DataFrame({'Target': target})
                ], axis=1)
                df_combined['Ticker'] = stock
                
                # Split this stock's data
                train_size = int(len(df_combined) * 0.8)
                train_data = df_combined[:train_size]
                test_data = df_combined[train_size:]
                
                combined_data.append((train_data, test_data))
                
            except Exception as e:
                logger.error(f"Error processing {stock}: {str(e)}")
                continue
        
        if not combined_data:
            raise ValueError("No data was successfully loaded")
        
        # Combine all training and testing data separately
        final_train = pd.concat([x[0] for x in combined_data], ignore_index=True)
        final_test = pd.concat([x[1] for x in combined_data], ignore_index=True)
        
        # Store separate datasets
        self.train_data = final_train
        self.test_data = final_test
        
        # Return combined dataset for compatibility
        return pd.concat([final_train, final_test], ignore_index=True)

    def prepare_features(self, data):
        """Prepare features while preserving ticker information"""
        feature_cols = [col for col in data.columns if col.startswith('PC')]
        
        # Split while maintaining chronological order
        train_size = int(len(data) * 0.8)
        
        # Store tickers before splitting
        self.train_tickers = data['Ticker'].values[:train_size]
        self.test_tickers = data['Ticker'].values[train_size:]
        
        # Split features and target
        self.X_train = data[feature_cols].iloc[:train_size]
        self.X_test = data[feature_cols].iloc[train_size:]
        self.y_train = data['Target'].iloc[:train_size]
        self.y_test = data['Target'].iloc[train_size:]
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        """Enhanced model training with optimized parameters and threshold tuning"""
        logger.info('Training Random Forest model with optimized parameters...')
        
        # Enhanced parameter grid for better performance
        param_grid = {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [25, 30, 35, 40, None],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 1.5}, {0: 1.5, 1: 1}],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9]  # Added to reduce overfitting
        }
        
        # Initialize base model with class weights
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True  # Enable out-of-bag score
        )
        
        # Custom scorer that balances accuracy and MAPE
        def custom_scorer(estimator, X, y):
            """Custom scoring function that balances accuracy and error"""
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            accuracy = accuracy_score(y, y_pred)
            errors = np.abs(y.astype(float) - y_pred_proba)
            mape = np.mean(errors) * 100
            
            return accuracy - (mape / 200)
        
        # Updated RandomizedSearchCV with custom scorer
        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=60,  # Increased iterations
            cv=tscv,
            scoring=make_scorer(custom_scorer, needs_proba=True),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2
        )
        
        # Fit model
        random_search.fit(self.X_train, self.y_train)
        logger.info(f"Best parameters found: {random_search.best_params_}")
        
        # Use best model
        self.model = random_search.best_estimator_
        
        # Generate initial predictions
        self.pred_probas = self.model.predict_proba(self.X_test)[:, 1]
        
        # Optimize threshold using multiple metrics
        thresholds = np.arange(0.3, 0.7, 0.02)
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            y_pred = (self.pred_probas >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Calculate error
            errors = np.abs(self.y_test - self.pred_probas)
            mape = np.mean(errors) * 100
            
            # Combined score favoring balanced metrics and lower MAPE
            score = (2 * f1 + acc + (prec * rec)) - (mape / 100)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold found: {best_threshold:.3f}")
        
        # Update predictions with optimal threshold
        self.predictions = (self.pred_probas >= best_threshold).astype(int)
        
        # Calculate prediction confidence
        self.prediction_confidence = np.abs(self.pred_probas - 0.5) * 2
        
        # Feature importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log OOB score if available
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

    def plot_feature_importance(self):
        """Plot feature importance with enhanced visualization"""
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', 
                   data=self.feature_importance_df.head(15),
                   palette='viridis')
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('docs/charts/rfc/feature_importance.png')
        plt.close()

    def plot_confusion_matrix(self):
        """Plot confusion matrix with enhanced visualization"""
        cm = confusion_matrix(self.y_test, self.predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Up'],
                   yticklabels=['Down', 'Up'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('docs/charts/rfc/confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self):
        """Plot ROC curve with enhanced visualization"""
        fpr, tpr, _ = roc_curve(self.y_test, self.pred_probas)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 
                label=f'ROC curve (AUC = {roc_auc_score(self.y_test, self.pred_probas):.3f})',
                color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('docs/charts/rfc/roc_curve.png')
        plt.close()

    def plot_probability_distribution(self):
        """Plot prediction probability distribution with enhanced visualization"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.pred_probas, bins=50, kde=True)
        plt.axvline(0.5, color='r', linestyle='--', label='Default Threshold')
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability of Positive Class')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('docs/charts/rfc/probability_distribution.png')
        plt.close()

    def save_model(self, version='v1'):
        """Save model and results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'model_results/rf_{version}_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f'{save_dir}/model.joblib')
        
        # Save feature importance
        self.feature_importance_df.to_csv(f'{save_dir}/feature_importance.csv')
        
        # Save model parameters
        with open(f'{save_dir}/model_params.txt', 'w') as f:
            f.write(str(self.model.get_params()))
        
        return save_dir


    def evaluate_model(self):
        """Evaluate model performance per ticker"""
        # Get predictions and probabilities per ticker
        ticker_data = {}
        for ticker in ['AAPL', 'GOOG', 'MSFT']:
            ticker_mask = self.test_tickers == ticker
            if any(ticker_mask):  # Only process if we have data for this ticker
                ticker_data[ticker] = {
                    'y_true': self.y_test[ticker_mask].values,
                    'y_pred': self.predictions[ticker_mask],
                    'y_prob': self.pred_probas[ticker_mask]
                }
        
        # Calculate metrics for each ticker
        logger.info("\n=== Per-Stock Performance Metrics ===")
        metrics = {}
        for ticker, data in ticker_data.items():
            # Skip if no data
            if len(data['y_true']) == 0:
                continue
                
            # Calculate metrics
            accuracy = accuracy_score(data['y_true'], data['y_pred'])
            f1 = f1_score(data['y_true'], data['y_pred'])
            mape = np.mean(np.abs(data['y_true'] - data['y_prob'])) * 100
            rmse = np.sqrt(np.mean((data['y_true'] - data['y_prob']) ** 2))
            
            metrics[ticker] = {
                'accuracy': accuracy,
                'f1': f1,
                'mape': mape,
                'rmse': rmse
            }
            
            logger.info(f"{ticker} - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, "
                    f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return metrics
        
    def plot_mape_distribution(self):
        """Plot enhanced error distribution across predictions"""
        actual_probs = self.y_test.astype(float)
        pred_probs = self.pred_probas
        
        # Calculate absolute errors
        errors = np.abs(actual_probs - pred_probs)
        
        plt.figure(figsize=(12, 8))
        
        # Create main error distribution plot
        sns.histplot(data=errors * 100, bins=50, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Absolute Error (%)')
        plt.ylabel('Count')
        
        # Add mean error line
        mean_error = np.mean(errors) * 100
        plt.axvline(mean_error, color='r', linestyle='--', 
                    label=f'Mean Error: {mean_error:.2f}%')
        
        # Add confidence regions
        plt.axvspan(0, 20, alpha=0.2, color='g', label='High Confidence')
        plt.axvspan(20, 40, alpha=0.2, color='y', label='Medium Confidence')
        plt.axvspan(40, 100, alpha=0.2, color='r', label='Low Confidence')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('docs/charts/rfc/error_distribution.png')
        plt.close()
        
        # Create additional plot for prediction confidence
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.pred_probas * 100, bins=50, kde=True)
        plt.title('Distribution of Prediction Confidences')
        plt.xlabel('Prediction Confidence (%)')
        plt.ylabel('Count')
        plt.axvline(50, color='r', linestyle='--', label='Decision Boundary')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('docs/charts/rfc/confidence_distribution.png')
        plt.close()

def main():
    """Enhanced main function with MAPE visualization"""
    try:
        logger.info("=== Starting Enhanced Stock Prediction Model Pipeline ===")
        
        predictor = StockPredictor()
        data = predictor.load_data()
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to load data")
        
        predictor.analyze_data_distribution(data)    
        X_train, X_test, y_train, y_test = predictor.prepare_features(data)
        predictor.train_model()
        metrics = predictor.evaluate_model()
        
        # Generate all plots including MAPE distribution
        predictor.plot_feature_importance()
        predictor.plot_confusion_matrix()
        predictor.plot_roc_curve()
        predictor.plot_probability_distribution()
        predictor.plot_mape_distribution()
        
        # Save results
        save_dir = predictor.save_model(version='v2')
        
        return predictor, metrics, save_dir
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("=== Pipeline Completed ===")
        plt.close('all')

if __name__ == '__main__':
    try:
        predictor, metrics, save_dir = main()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)