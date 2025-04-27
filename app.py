from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model
model_file = 'synthetictraffic_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Synthetic Traffic Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.json
        features = data['features']  # Expecting a list of features
        
        # Ensure the input matches the model's expected shape
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
