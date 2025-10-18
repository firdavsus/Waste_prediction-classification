import csv
import pandas as pd

# === 1. Load data ===
def get_weight_csv(path='ready_data/full.csv'):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return pd.DataFrame(data)

df = get_weight_csv()

numbers = ['Konteinerio_volume', 'temperature_2m_mean (В°C)', 'wind_speed_10m_max (km/h)', 'snowfall_sum (cm)', 'rain_sum (mm)']
emb = ['Konteinerio_area', 'Konteinerio_street', 'Driver', 'Weather_code']
position = ['Konteinerio_latitude', 'Konteinerio_longitude']

# predict = ['Planned departure date']
# there are more than 1 planned departure date for on 'Konteinerio_Nr' and based on this I want to predict the time it takes to empty it
# there we do not have the weight so we will assume if the irst date is 7th and second 22th it means that durting 15 days container
# will be full, so model shold predict the time it takes for container to gte full (like max weight)

# in some Konatine_Nr there ven 2-6 (or more, if less skip) entries which means we can preict from 1 time till 5 times using rnn is good like adjustable length!

for col in numbers + position:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=numbers + position, inplace=True)

from sklearn.preprocessing import StandardScaler, LabelEncoder

label_encoders = {}
for col in emb:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
df[numbers + position] = scaler.fit_transform(df[numbers + position])

# now I should get the time it took to replace it if no further it will be None/or dropped if only entriw with this id

