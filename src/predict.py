import numpy as np
import joblib
from keras.models import load_model

# Load the trained model and scaler, and the last window
model_path = '../models/gru_model.keras'
scaler_path = '../data/processed/scaler.pkl'
last_window_path = '../data/processed/last_window.npy'

model = load_model(model_path)
scaler = joblib.load(scaler_path)
last_window = np.load(last_window_path)

def predict_next_n_days(model, scaler, last_window, n_days):
    predictions = []
    current_input = last_window
    
    for _ in range(n_days):
        prediction_scaled = model.predict(current_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        predictions.append(prediction[0][0])
        
        # Reshape prediction_scaled to have the same shape as current_input[:, -1:, :]
        prediction_scaled = np.reshape(prediction_scaled, (1, 1, 1))
        
        # Append the predicted value to the current input and shift
        new_input = np.append(current_input[:, 1:, :], prediction_scaled, axis=1)
        current_input = new_input
    
    return predictions

if __name__ == "__main__":
    # Predict stock prices for the next N days
    n_days = int(input("Enter the number of days for prediction: "))
    
    # Load the last window of data from a saved file
    last_window = last_window

    predictions = predict_next_n_days(model, scaler, last_window, n_days)

    print(f"Predicted stock prices for the next {n_days} days:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: {pred}")