import psycopg2
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = "churn_model.pkl"
model = joblib.load(MODEL_PATH)

# Connect to Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@db:5432/predictions_db")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# ✅ Ensure the table is created at startup
def create_table():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            features TEXT,
            prediction INT
        )
    """)
    conn.commit()

create_table()  # Call this function at app startup

# ✅ Serve the Home Page UI
@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html exists in "templates" folder

# ✅ Handle Predictions (Supports Both GET and POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Send a POST request with data to get predictions."})

    try:
        features = [
            float(request.form["account_length"]),
            float(request.form["num_vmail_messages"]),
            float(request.form["total_day_calls"]),
            float(request.form["total_day_charge"]),
            float(request.form["total_eve_calls"]),
            float(request.form["total_eve_charge"]),
            float(request.form["total_night_calls"]),
            float(request.form["total_night_charge"]),
            float(request.form["total_intl_calls"]),
            float(request.form["total_intl_charge"]),
            float(request.form["customer_service_calls"]),
            int(request.form["international_plan"]),
            int(request.form["voice_mail_plan"])
        ]

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        # Store prediction in database
        cursor.execute("INSERT INTO predictions (features, prediction) VALUES (%s, %s)", (str(features), int(prediction)))
        conn.commit()

        return render_template("index.html", prediction_text=f"Prediction: {int(prediction)}")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)
