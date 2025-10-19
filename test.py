import requests

url = "http://127.0.0.1:8000/predict-weight"
data = {
    "lift_point_latitude": 54.7518,
    "lift_point_longitude": 25.2415,
    "Konteinerio_capacity": 1.1,
    "people_on_street_nearest_avg3": 38,
    "season": "Summer",
    "Zona": "Vilniaus m. BA1",
    "Driver": "Ekonovus",
    "Konteinerio_street": "Ožiaragio g."
}

response = requests.post(url, data=data)
print(response.json())

url = "http://127.0.0.1:8000/predict-days"
data = {
    "Konteinerio_area": "Vilniaus m.",
    "Konteinerio_street": "Žėručio g.",
    "Driver": "Ecoservice",
    "Weather_code": "3.0",
    "Konteinerio_volume": 1.1,
    "temperature_2m_mean": 10,
    "wind_speed_10m_max": 5,
    "snowfall_sum": 0,
    "rain_sum": 0,
    "Konteinerio_latitude": 54.675,
    "Konteinerio_longitude": 25.209
}

response = requests.post(url, data=data)
print(response.json())

url = "http://127.0.0.1:8000/detect-image"
files = {"image": open("example2.jpg", "rb")}

response = requests.post(url, files=files)
with open("detected.png", "wb") as f:
    f.write(response.content)
print("Saved detected image as detected.png")
