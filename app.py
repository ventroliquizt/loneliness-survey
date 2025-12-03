from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model + preprocessor
# -----------------------------
MODEL_FILENAME = '/home/vent/vscode/vscode./ML/project/comprehensive_predictor_model.pkl'
PREPROCESSOR_FILENAME = '/home/vent/vscode/vscode./ML/project/comprehensive_preprocessor.pkl'

with open(MODEL_FILENAME, 'rb') as f:
    model = pickle.load(f)

with open(PREPROCESSOR_FILENAME, 'rb') as f:
    preprocessor = pickle.load(f)

print("Model + Preprocessor Loaded!")


# -----------------------------
# SAME order of features used in training
# -----------------------------
feature_names = [
    "Lack_Comp", "Left_Out", "Isolated", "Talk_Friends", "Understood",
    "Rely_On", "Unknown_Me", "Ask_Help", "Meaningful_Rels_R",
    "Avoid_Money", "In_Person_Meetings", "Time_Outside_Home",
    "Weekend_Social", "Phone_Hours", "Phone_Late_Night",
    "Sleep_Hours", "Sleep_Restful", "Physical_Activity",
    "Depressed", "Little_Interest", "Year", "Living"
]


# -----------------------------
# Prediction helper (from your predict.py)
# -----------------------------
def get_prediction(raw_data: dict):

    df_new = pd.DataFrame([raw_data])

    # Convert Meaningful_Rels → Meaningful_Rels_R
    if 'Meaningful_Rels' in df_new:
        df_new['Meaningful_Rels_R'] = 6 - df_new['Meaningful_Rels']
        df_new.drop('Meaningful_Rels', axis=1, inplace=True)

    # Transform data using preprocessor
    X_new_processed = preprocessor.transform(df_new)

    # Predict label + probability
    label = int(model.predict(X_new_processed)[0])
    prob = float(model.predict_proba(X_new_processed)[0][1])

    return label, prob


# -----------------------------
# API ROUTES
# -----------------------------
@app.route('/')
def home():
    return "Loneliness Predictor API Running"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    label, prob = get_prediction(data)

    return jsonify({
        'loneliness_level': label,
        'probability': prob
    })


# -----------------------------
# Run API
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
