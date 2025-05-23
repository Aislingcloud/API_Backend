# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained classification model and label encoder
cls_model = joblib.load('trained_data/model_cls.pkl')
label_encoder = joblib.load('trained_data/label_encoder.pkl')

# List of required features
required_features = ['Total_Score', 'Midterm_Score', 'Final_Score', 'Projects_Score', 'Attendance (%)']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Check if all required fields are provided
    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Prepare the input data
    input_df = pd.DataFrame([[data[feature] for feature in required_features]], columns=required_features)

    # Predict using the model
    prediction_encoded = cls_model.predict(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Respond with prediction
    return jsonify({
        "prediction": prediction_label
    })

if __name__ == '__main__':
    app.run(debug=True)
