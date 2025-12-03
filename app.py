from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model + preprocessor
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = os.path.join(BASE_DIR, "comprehensive_predictor_model.pkl")
PREPROCESSOR_FILENAME = os.path.join(BASE_DIR, "comprehensive_preprocessor.pkl")

with open(MODEL_FILENAME, 'rb') as f:
    model = pickle.load(f)

with open(PREPROCESSOR_FILENAME, 'rb') as f:
    preprocessor = pickle.load(f)

print("Model + Preprocessor Loaded!")


feature_names = [
    "Lack_Comp", "Left_Out", "Isolated", "Talk_Friends", "Understood",
    "Rely_On", "Unknown_Me", "Ask_Help", "Meaningful_Rels_R",
    "Avoid_Money", "In_Person_Meetings", "Time_Outside_Home",
    "Weekend_Social", "Phone_Hours", "Phone_Late_Night",
    "Sleep_Hours", "Sleep_Restful", "Physical_Activity",
    "Depressed", "Little_Interest", "Year", "Living"
]


def get_prediction(raw_data: dict):
    df_new = pd.DataFrame([raw_data])

    if 'Meaningful_Rels' in df_new:
        df_new['Meaningful_Rels_R'] = 6 - df_new['Meaningful_Rels']
        df_new.drop('Meaningful_Rels', axis=1, inplace=True)

    X_new_processed = preprocessor.transform(df_new)

    label = int(model.predict(X_new_processed)[0])
    prob = float(model.predict_proba(X_new_processed)[0][1])

    return label, prob


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    label, prob = get_prediction(data)
    return jsonify({
        'loneliness_level': label,
        'probability': prob
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
