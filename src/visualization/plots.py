import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, stock_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'LSTM Predictions vs Actual for {stock_name}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'../docs/charts/lstm/predictions_{stock_name}.png')
    plt.close()

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../docs/charts/lstm/training_history.png')
    plt.close()