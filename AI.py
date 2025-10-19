from Image.image import detect_image
from weaght_of_bin.weight_predict_apply import GetWeight
from days_of_bin.days_model import WasteLSTMPredictor
import pandas as pd

# first function for image->image
detect_image("Image/inputs/example1.jpg", "Image/outputs/", conf=0.5, model_path="Image/weights/yoloooo.pt")

# # get the weight of a certain input
get_it = GetWeight(artifacts_path = "weaght_of_bin/artifacts.pkl", model_path="weaght_of_bin/weight_model.pth")
input_dict = {
            'lift_point_latitude': 54.7518,
            'Lift_point_longitude': 25.2415,
            'Konteinerio_capacity': 1.1,
            'people_on_street_nearest_avg3': 38,
            'season': 'Summer',
            'Zona': 'Vilniaus m. BA1',
            'Driver': 'Ekonovus',
            'Konteinerio_street': 'Ožiaragio g.'#'OЕѕiaragio g.'
        }
pred_weight = get_it.predict(input_dict)
print(f"Predicted weight: {pred_weight:.2f} kg")

# get the days till fullness
get_days = WasteLSTMPredictor(artifacts_path="days_of_bin/waste_lstm_artifacts.pkl")
new_data = pd.DataFrame({
        "Konteinerio_area": ["Vilniaus m."]*4,
        "Konteinerio_street": ["Žėručio g."]*4,
        "Driver": ["Ecoservice"]*4,
        "Weather_code": ['3.0', '51.0', '3.0', '3.0'],
        "Konteinerio_volume": [1.1]*4,
        "temperature_2m_mean (В°C)": [10.9, 10.5, 9.7, 11.5],
        "wind_speed_10m_max (km/h)": [14.4, 20.2, 11.2, 15],
        "snowfall_sum (cm)": [0, 0, 0, 0],
        "rain_sum (mm)": [0, 0.4, 0, 0],
        "Konteinerio_latitude": [54.675]*4,
        "Konteinerio_longitude": [25.209]*4
    })

pred_days = get_days.predict(new_data)
print("Predicted days:", pred_days)