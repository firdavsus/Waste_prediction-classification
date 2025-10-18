# import csv

# # location = ['lift_point_latitude', 'Lift_point_longitude']
# # numeric = ['Konteinerio_capacity', 'people_on_street_nearest_avg3']
# # emb=['season', 'Zona', 'Driver', 'Konteinerio_street']
# # predict = ['Weight_kg_']

# def get_weight():
#     data = []
#     with open('ready_data/lifts_with_people.csv', 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             data.append(row)
#     return data
        
        
        
# data=get_weight()


import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# === 1. Load data ===
def get_weight_csv(path='ready_data/lifts_with_people.csv'):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return pd.DataFrame(data)

df = get_weight_csv()

# === 2. Define features ===
coord_features = ['lift_point_latitude', 'Lift_point_longitude']
numeric_features = ['Konteinerio_capacity', 'people_on_street_nearest_avg3']
categorical_features = ['season', 'Zona', 'Driver', 'Konteinerio_street']
target_col = 'Weight_kg_'

# Convert numeric features to float
all_numeric_cols = coord_features + numeric_features + [target_col]

# Replace empty strings and invalid values with NaN
df[all_numeric_cols] = df[all_numeric_cols].replace(r'^\s*$', np.nan, regex=True)
for col in coord_features + numeric_features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=coord_features + numeric_features + [target_col], inplace=True)

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize numeric features
scaler = StandardScaler()
df[numeric_features + coord_features] = scaler.fit_transform(df[numeric_features + coord_features])

target_scaler = StandardScaler()
df[target_col] = target_scaler.fit_transform(df[[target_col]])

# === 3. Dataset class ===
class WasteDataset(Dataset):
    def __init__(self, df, coord_features, numeric_features, cat_features, target):
        self.coords = df[coord_features].values.astype(np.float32)
        self.numeric = df[numeric_features].values.astype(np.float32)
        self.cats = df[cat_features].values.astype(np.int64)
        self.target = df[target].values.astype(np.float32).reshape(-1,1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'coords': torch.tensor(self.coords[idx]),
            'numeric': torch.tensor(self.numeric[idx]),
            'cats': torch.tensor(self.cats[idx]),
            'target': torch.tensor(self.target[idx])
        }

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_ds = WasteDataset(train_df, coord_features, numeric_features, categorical_features, target_col)
val_ds = WasteDataset(val_df, coord_features, numeric_features, categorical_features, target_col)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# === 4. Model ===
class CoordEmbed(nn.Module):
    def __init__(self, emb_size=8, hidden_dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(2)  # normalizes per-sample across features
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_size)
        )

    def forward(self, coords):
        coords = self.norm(coords)
        return self.net(coords)

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class WasteWeightModel(nn.Module):
    def __init__(self, num_numeric, cat_dims, emb_size=8, hidden_dim=64, hid_for_pos=32, depth=4, dropout=0.20):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cat_dim, emb_size) for cat_dim in cat_dims])
        self.num_emb = emb_size * len(cat_dims)

        # Coordinate embedding MLP
        self.coord_embed = CoordEmbed(emb_size=emb_size, hidden_dim=hid_for_pos)

        # Input dimension
        self.coord_numeric_dim = emb_size + num_numeric
        input_dim = self.num_emb + self.coord_numeric_dim

        # Stack multiple hidden layers dynamically
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, coords, numeric, cats):
        cat_emb = [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_emb = torch.cat(cat_emb, dim=1)
        coord_emb = self.coord_embed(coords)

        x = torch.cat([cat_emb, coord_emb, numeric], dim=1)
        x = self.mlp(x)
        return self.out(x)


# Determine number of categories per categorical feature
cat_dims = [int(df[col].max())+1 for col in categorical_features]

model = WasteWeightModel(num_numeric=len(numeric_features),
                         cat_dims=cat_dims,
                         emb_size=64,
                         hidden_dim=64,
                         hid_for_pos=32,
                         depth=2)

# 64-64-32-4 -> 0.015, 128-64-64-4 -> 0.012, 32-32-16-4 -> 0.014
# 0.013                0.013            0.014

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === 5. Training setup ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.SmoothL1Loss()
epochs = 40

# === 6. Training loop ===
for epoch in range(epochs):
    model.train()
    train_loss = 0
    accum_steps = 32
    for i, batch in enumerate(train_loader):
        coords = batch['coords'].to(device)
        numeric = batch['numeric'].to(device)
        cats = batch['cats'].to(device)
        target = batch['target'].to(device)

        output = model(coords, numeric, cats)
        loss = criterion(output, target) / accum_steps
        loss.backward()

        train_loss += criterion(output, target).item() * coords.size(0)

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            coords = batch['coords'].to(device)
            numeric = batch['numeric'].to(device)
            cats = batch['cats'].to(device)
            target = batch['target'].to(device)
            output = model(coords, numeric, cats)
            val_loss += criterion(output, target).item() * coords.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# === 7. Predict on full dataset ===
model.eval()
full_ds = WasteDataset(df, coord_features, numeric_features, categorical_features, target_col)
full_loader = DataLoader(full_ds, batch_size=64)
preds = []
with torch.no_grad():
    for batch in full_loader:
        coords = batch['coords'].to(device)
        numeric = batch['numeric'].to(device)
        cats = batch['cats'].to(device)
        output = model(coords, numeric, cats)
        preds.append(output.cpu().numpy())
preds = np.vstack(preds)
preds_real = target_scaler.inverse_transform(preds)
true_weights = df[target_col].values.reshape(-1,1) 
true_weights_real = target_scaler.inverse_transform(true_weights)

avg_diff = 0
for i in range(100):
    print(f"Predicted: {preds_real[i][0]:.3f} kg, True: {true_weights_real[i][0]:.3f} kg")
    avg_diff += abs(preds_real[i][0] - true_weights_real[i][0])
avg_diff /= 100
print(f"Average difference: {avg_diff:.3f} kg")

