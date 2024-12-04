import pandas as pd
import matplotlib.pyplot as plt


def create_chart(model_name, companies):
    for company in companies:
        df = pd.read_csv(f'results/{model_name}_{company}_predictions.csv')
        df['Date'] = df['Date'] - 805
        df['Actual_sum'] = df['Actual'].cumsum()
        df['Predicted_sum'] = df['Predicted'].cumsum()

        #corrected_df = sum_df(df)
        plt.figure(figsize=(20, 6)) 
        plt.plot(df['Date'], df['Actual_sum'], label='Actual', color='blue') 
        plt.plot(df['Date'], df['Predicted_sum'], label='Model Prediction', color='red')

        plt.title(f'Accuracy of {model} for {company}') 
        plt.xlabel('Date')
        plt.ylabel('Number of Positive Market Days')
        plt.xticks(rotation=45)
        plt.legend()

        plt.savefig(f'docs/charts/{model}_{company}_chart.png', dpi=300, bbox_inches='tight')
        print(f'Saved chart for {model}')

if __name__ == '__main__':
    companies = ['AAPL', 'GOOG', 'MSFT']
    models = ['SVM', 'LSTM']

    print('Creating charts...')
    for model in models:
        create_chart(model, companies)

    print('Done!')