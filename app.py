from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("model/lasso_cv.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Feature Order (VERY IMPORTANT)
feature_order = [
    "temperature_celsius",
    "humidity_percent",
    "household_size",
    "ac_usage_hours"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        input_data = [float(request.form[feature]) for feature in feature_order]

        # Convert to numpy array
        input_array = np.array([input_data])

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Electricity Units (kWh): {round(prediction[0], 2)}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text="Error in input data.")

if __name__ == "__main__":
    app.run(debug=True)
