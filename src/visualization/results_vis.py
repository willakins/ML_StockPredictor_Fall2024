import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class model_plotter:
    def __init__(self, df, model, company):
        self.df = df
        self.model = model
        self.company = company
    
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

    def plot_accuracy(self):
        plt.figure(figsize=(20, 6)) 
        plt.plot(self.df['Date'], self.df['Actual_sum'], label='Actual', color='blue') 
        plt.plot(self.df['Date'], self.df['Predicted_sum'], label='Model Prediction', color='red')

        plt.title(f'Accuracy of {self.model} for {self.company}') 
        plt.xlabel('Date')
        plt.ylabel('Number of Positive Market Days')
        plt.xticks(rotation=45)
        plt.legend()

        plt.savefig(f'docs/charts/{self.model}_{self.company}_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_chart(model_name, companies):
    for company in companies:
        df = pd.read_csv(f'results/{model_name}_{company}_predictions.csv')
        df['Date'] = df['Date'] - 805
        df['Actual_sum'] = df['Actual'].cumsum()
        df['Predicted_sum'] = df['Predicted'].cumsum()

        plotter = model_plotter(df, model_name, company)
        plotter.plot_accuracy()

        print(f'Saved chart for {model}')

if __name__ == '__main__':
    companies = ['AAPL', 'GOOG', 'MSFT']
    models = ['SVM', 'LSTM']

    print('Creating charts...')
    for model in models:
        create_chart(model, companies)

    print('Done!')