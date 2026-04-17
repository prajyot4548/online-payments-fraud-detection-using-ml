from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

from utils import create_features

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================

model = pickle.load(open("model/fraud_model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# =========================
# RULE ENGINE (VERY IMPORTANT)
# =========================

def rule_check(data):
    # Rule 1: Balance mismatch
    if abs(data["oldbalanceOrg"] - data["newbalanceOrig"] - data["amount"]) > 1:
        return 1

    # Rule 2: Empty destination + high amount
    if data["oldbalanceDest"] == 0 and data["amount"] > 50000:
        return 1

    # Rule 3: full balance transfer
    if data["oldbalanceOrg"] > 0 and data["newbalanceOrig"] == 0:
        return 1

    return 0

# =========================
# PREPROCESS (FIXED)
# =========================

def preprocess(data_dict):
    df = pd.DataFrame([data_dict])

    # Feature engineering
    df = create_features(df)

    # Match training columns
    df = df[columns]

    # Scale
    return scaler.transform(df)

# =========================
# HOME
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# API (FIXED + SMART LOGIC)
# =========================

@app.route("/api/predict", methods=["POST"])
def api():
    try:
        data = request.json

        # Encode type
        type_encoded = encoder.transform([data["type"]])[0]

        # Create proper input dictionary (IMPORTANT FIX)
        input_dict = {
            "step": float(data.get("step", 0)),
            "type": type_encoded,
            "amount": float(data.get("amount", 0)),
            "oldbalanceOrg": float(data.get("oldbalanceOrg", 0)),
            "newbalanceOrig": float(data.get("newbalanceOrig", 0)),
            "oldbalanceDest": float(data.get("oldbalanceDest", 0)),
            "newbalanceDest": float(data.get("newbalanceDest", 0)),
            "isFlaggedFraud": 0
        }

        # 🔥 RULE CHECK FIRST
        rule_result = rule_check(input_dict)

        # ML prediction
        processed = preprocess(input_dict)
        prob = model.predict_proba(processed)[0][1]

        # 🔥 FINAL DECISION (RULE + ML)
        if rule_result == 1:
            final_pred = 1
            prob = max(prob, 0.85)  # boost confidence

        else:
            final_pred = 1 if prob >= 0.5 else 0

        risk = round(prob * 100, 2)

        # =========================
        # RISK LEVEL
        # =========================

        if risk >= 70:
            level = "HIGH"
            result = "Fraud"
            status = "fraud"

        elif risk >= 40:
            level = "MEDIUM"
            result = "Risk"
            status = "fraud"

        else:
            level = "LOW"
            result = "Safe"
            status = "safe"

        return jsonify({
            "prediction": result,
            "risk": risk,
            "level": level,
            "status": status
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})

# =========================

if __name__ == "__main__":
    app.run(debug=True)