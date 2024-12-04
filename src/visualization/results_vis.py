import pandas as pd
import matplotlib.pyplot as plt
# Step 1: Read data from CSV file
def create_chart(model_name, companies):
    for company in companies:
        df = pd.read_csv(f'{model_name}_{company}_predictions.csv')

        plt.figure(figsize=(10, 6)) 
        plt.plot(df['Date'], df['Actual'], label='Actual', color='blue') 
        plt.plot(df['Date'], df['Predicted'], label='Model Prediction', color='red')

        plt.title(f'Accuracy of {model} for {company}') 
        plt.xlabel('Date')
        plt.ylabel('Market will go up')
        plt.xticks(rotation=45)
        plt.legend()

        plt.savefig(f'{model}_{company}_chart.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    companies = ['AAPL', 'GOOG', 'MSFT']
    models = ['svm', 'lstm']

    for model in models:
        create_chart(model, companies)