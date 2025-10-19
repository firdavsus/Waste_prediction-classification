from flask import Flask, request, jsonify, send_file
from io import BytesIO
import pandas as pd
from Image.image import detect_image_bytes
from weaght_of_bin.weight_predict_apply import GetWeight
from days_of_bin.days_model import WasteLSTMPredictor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # this 

# ======================= INIT MODELS =======================
weight_model = GetWeight("weaght_of_bin/artifacts.pkl", "weaght_of_bin/weight_model.pth")
days_model = WasteLSTMPredictor("days_of_bin/waste_lstm_artifacts.pkl")


# ======================= ENDPOINTS ========================
@app.route("/predict-weight", methods=["POST"])
def predict_weight():
    data = request.form
    input_dict = {
        'lift_point_latitude': float(data['lift_point_latitude']),
        'Lift_point_longitude': float(data['lift_point_longitude']),
        'Konteinerio_capacity': float(data['Konteinerio_capacity']),
        'people_on_street_nearest_avg3': int(data['people_on_street_nearest_avg3']),
        'season': data['season'],
        'Zona': data['Zona'],
        'Driver': data['Driver'],
        'Konteinerio_street': data['Konteinerio_street']
    }
    pred_weight = weight_model.predict(input_dict)
    return jsonify({"predicted_weight_kg": float(pred_weight)})


@app.route("/predict-days", methods=["POST"])
def predict_days():
    data = request.form
    df = pd.DataFrame([{
        "Konteinerio_area": data['Konteinerio_area'],
        "Konteinerio_street": data['Konteinerio_street'],
        "Driver": data['Driver'],
        "Weather_code": data['Weather_code'],
        "Konteinerio_volume": float(data['Konteinerio_volume']),
        "temperature_2m_mean (В°C)": float(data['temperature_2m_mean']),
        "wind_speed_10m_max (km/h)": float(data['wind_speed_10m_max']),
        "snowfall_sum (cm)": float(data['snowfall_sum']),
        "rain_sum (mm)": float(data['rain_sum']),
        "Konteinerio_latitude": float(data['Konteinerio_latitude']),
        "Konteinerio_longitude": float(data['Konteinerio_longitude'])
    }])
    pred_days = days_model.predict(df)
    return jsonify({"predicted_days": pred_days.tolist()})


@app.route("/detect-image", methods=["POST"])
def detect_image_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    output_bytes = detect_image_bytes(image_bytes, conf=0.35, model_path="Image/weights/yoloooo.pt")
    
    return send_file(BytesIO(output_bytes), mimetype='image/png', as_attachment=False)


# ======================= RUN SERVER =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
