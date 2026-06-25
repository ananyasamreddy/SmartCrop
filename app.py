from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

df = pd.read_excel("crop_dataset.xlsx")
df.columns = df.columns.str.strip().str.lower()
df["soil_type"] = df["soil_type"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

soil_encoder = LabelEncoder()
df["soil_type_encoded"] = soil_encoder.fit_transform(df["soil_type"])

X = df[["temperature", "humidity", "rainfall", "soil_type_encoded"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)


def normalize_crop_name(name):
    return str(name).strip().lower().replace(" ", "")

crop_support_data = {
    "rice": {"fertilizers": ["Urea", "DAP", "Potash"], "season": "Kharif", "duration": "110-130 days", "water_requirement": "High", "ideal_soil": "Clayey Soil"},
    "maize": {"fertilizers": ["Urea", "NPK", "Compost"], "season": "Kharif/Rabi", "duration": "90-110 days", "water_requirement": "Moderate", "ideal_soil": "Loamy Soil"},
    "chickpea": {"fertilizers": ["Super Phosphate", "Compost", "Potash"], "season": "Rabi", "duration": "100-120 days", "water_requirement": "Low to Moderate", "ideal_soil": "Loamy Soil"},
    "kidneybeans": {"fertilizers": ["DAP", "Farmyard Manure", "Potash"], "season": "Rabi", "duration": "90-100 days", "water_requirement": "Moderate", "ideal_soil": "Loamy Soil"},
    "pigeonpeas": {"fertilizers": ["DAP", "Compost", "Potash"], "season": "Kharif", "duration": "120-180 days", "water_requirement": "Moderate", "ideal_soil": "Red Soil"},
    "mothbeans": {"fertilizers": ["Organic Compost", "DAP", "Potash"], "season": "Kharif", "duration": "75-90 days", "water_requirement": "Low", "ideal_soil": "Sandy Soil"},
    "mungbean": {"fertilizers": ["DAP", "Compost", "Biofertilizer"], "season": "Kharif/Zaid", "duration": "60-75 days", "water_requirement": "Low to Moderate", "ideal_soil": "Loamy Soil"},
    "blackgram": {"fertilizers": ["DAP", "Organic Manure", "Potash"], "season": "Kharif", "duration": "70-90 days", "water_requirement": "Moderate", "ideal_soil": "Black Soil"},
    "lentil": {"fertilizers": ["Super Phosphate", "Compost", "Potash"], "season": "Rabi", "duration": "100-110 days", "water_requirement": "Low", "ideal_soil": "Loamy Soil"},
    "pomegranate": {"fertilizers": ["Organic Manure", "NPK", "Micronutrients"], "season": "Annual", "duration": "150-180 days", "water_requirement": "Moderate", "ideal_soil": "Black Soil"},
    "banana": {"fertilizers": ["Urea", "Potash", "Organic Manure"], "season": "Annual", "duration": "11-12 months", "water_requirement": "High", "ideal_soil": "Loamy Soil"},
    "mango": {"fertilizers": ["FYM", "NPK", "Micronutrients"], "season": "Annual", "duration": "4-5 months fruiting cycle", "water_requirement": "Moderate", "ideal_soil": "Alluvial Soil"},
    "grapes": {"fertilizers": ["NPK", "Organic Compost", "Potash"], "season": "Annual", "duration": "120-140 days", "water_requirement": "Moderate", "ideal_soil": "Loamy Soil"},
    "watermelon": {"fertilizers": ["NPK", "Compost", "Potash"], "season": "Zaid", "duration": "80-100 days", "water_requirement": "Moderate", "ideal_soil": "Sandy Soil"},
    "muskmelon": {"fertilizers": ["Compost", "NPK", "Potash"], "season": "Zaid", "duration": "75-90 days", "water_requirement": "Moderate", "ideal_soil": "Sandy Soil"},
    "apple": {"fertilizers": ["Organic Manure", "NPK", "Calcium"], "season": "Temperate", "duration": "150-180 days", "water_requirement": "Moderate", "ideal_soil": "Loamy Soil"},
    "orange": {"fertilizers": ["NPK", "Organic Manure", "Micronutrients"], "season": "Annual", "duration": "6-8 months fruiting cycle", "water_requirement": "Moderate", "ideal_soil": "Red Soil"},
    "papaya": {"fertilizers": ["Urea", "Super Phosphate", "Potash"], "season": "Annual", "duration": "8-10 months", "water_requirement": "Moderate", "ideal_soil": "Loamy Soil"},
    "coconut": {"fertilizers": ["Organic Manure", "NPK", "Magnesium"], "season": "Annual", "duration": "Year-round", "water_requirement": "High", "ideal_soil": "Sandy Soil"},
    "cotton": {"fertilizers": ["Urea", "DAP", "Potash"], "season": "Kharif", "duration": "150-180 days", "water_requirement": "Moderate", "ideal_soil": "Black Soil"},
    "jute": {"fertilizers": ["Urea", "Super Phosphate", "Potash"], "season": "Kharif", "duration": "120-150 days", "water_requirement": "High", "ideal_soil": "Alluvial Soil"},
    "coffee": {"fertilizers": ["Organic Manure", "NPK", "Lime"], "season": "Perennial", "duration": "3-4 years to mature", "water_requirement": "Moderate", "ideal_soil": "Laterite Soil"}
}


def get_market_price(crop_name):
    if "market_price" not in df.columns:
        return "Not Available"
    crop_rows = df[df["label"].str.strip().str.lower() == crop_name.strip().lower()]
    if not crop_rows.empty and "market_price" in crop_rows:
        return int(crop_rows["market_price"].mode()[0])
    return "Not Available"


def get_weather_tip(temperature, humidity, rainfall):
    tips = []
    if rainfall > 200:
        tips.append("High rainfall detected. Ensure proper drainage.")
    elif rainfall < 50:
        tips.append("Low rainfall detected. Plan irrigation.")

    if temperature > 32:
        tips.append("High temperature detected. Use mulching.")
    elif temperature < 18:
        tips.append("Cool weather detected. Growth may slow.")

    if humidity < 50:
        tips.append("Low humidity detected. Monitor moisture.")
    elif humidity > 85:
        tips.append("High humidity detected. Watch for diseases.")

    if not tips:
        tips.append("Conditions are suitable for crop growth.")
    return " ".join(tips)


def get_crop_support(crop_name, selected_soil):
    key = normalize_crop_name(crop_name)
    details = crop_support_data.get(key, {})
    ideal_soil = details.get("ideal_soil", selected_soil)
    if selected_soil == ideal_soil:
        soil_message = f"{selected_soil} is highly suitable for {crop_name}."
    else:
        soil_message = f"{selected_soil} is moderately suitable. Ideal: {ideal_soil}."
    return {
        "soil_message": soil_message,
        "fertilizers": details.get("fertilizers", ["Organic Compost"]),
        "crop_details": details
    }


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def serve_files(filename):
    return send_from_directory(".", filename)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])
        soil = data["soil"].strip()

        errors = []

        if not (8.8 <= temperature <= 43.6):
            errors.append("temperature")
        if not (14.2 <= humidity <= 99.9):
            errors.append("humidity")      
        if not (20.2 <= rainfall <= 298.5):
            errors.append("rainfall")
        if soil not in soil_encoder.classes_:
            errors.append("soil")

        if errors:
            return jsonify({"message": f"Invalid {', '.join(errors)}"}), 400

        soil_encoded = int(soil_encoder.transform([soil])[0])
        input_row = [[temperature, humidity, rainfall, soil_encoded]]

        probabilities = model.predict_proba(input_row)[0]
        class_names = model.classes_

        prob_pairs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
        if prob_pairs[0][1] < 0.3:
            return jsonify({"crop": "None", "confidence": "0%", "message": "No suitable crop found"})

        best_crop = str(prob_pairs[0][0])

        top3 = [
            {"crop": str(name), "market_price": get_market_price(name), "score": int(round(prob * 100))}
            for name, prob in prob_pairs[:3]
        ]

        support = get_crop_support(best_crop, soil)

        return jsonify({
            "crop": best_crop,
            "confidence": int(round(prob_pairs[0][1] * 100)),
            "market_price": get_market_price(best_crop),
            "top3": top3,
            "soil_message": support["soil_message"],
            "weather_tip": get_weather_tip(temperature, humidity, rainfall),
            "fertilizers": support["fertilizers"],
            "crop_details": support["crop_details"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
