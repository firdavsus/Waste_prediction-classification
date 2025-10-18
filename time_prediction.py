import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

# === Load CSV ===
df = pd.read_csv("ready_data/full.csv", parse_dates=['Planned departure date'])

# Keep containers with ≥2 entries
counts = df['Konteinerio_Nr'].value_counts()
keep_ids = counts[counts >= 2].index
df = df[df['Konteinerio_Nr'].isin(keep_ids)].copy()

# Sort by container and date
df = df.sort_values(['Konteinerio_Nr', 'Planned departure date']).reset_index(drop=True)

# Compute target_days
df['target_days'] = df.groupby('Konteinerio_Nr')['Planned departure date'].shift(-1) - df['Planned departure date']
df['target_days'] = df['target_days'].dt.total_seconds() / (24*3600)  # days
df = df.dropna(subset=['target_days']).reset_index(drop=True)
df['target_days_log'] = np.log1p(df['target_days'])  # log(1 + days)

# Encode categorical and scale numeric features
categorical_cols = ['Konteinerio_area', 'Konteinerio_street', 'Driver', 'Weather_code']
numeric_cols = ['Konteinerio_volume', 'temperature_2m_mean (В°C)', 
                'wind_speed_10m_max (km/h)', 'snowfall_sum (cm)', 'rain_sum (mm)', 
                'Konteinerio_latitude', 'Konteinerio_longitude']

label_encoders = {}
cat_dims = []
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    cat_dims.append(len(le.classes_))

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === Group sequences by container ===
grouped = df.groupby('Konteinerio_Nr')
sequences = []
targets = []

for _, group in grouped:
    features = group[categorical_cols + numeric_cols].values.astype(np.float32)
    target = group['target_days_log'].values.astype(np.float32)
    sequences.append(features)
    targets.append(target)

# === DEBUG: Check feature dimensions ===
print(f"Total features: {len(categorical_cols) + len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

# === PyTorch Dataset ===
class WasteLSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'target': torch.tensor(self.targets[idx])
        }

dataset = WasteLSTMDataset(sequences, targets)

# === Collate function ===
def collate_fn(batch):
    seqs = [b['sequence'] for b in batch]
    tars = [b['target'] for b in batch]
    lengths = torch.tensor([len(s) for s in seqs])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    padded_tars = torch.nn.utils.rnn.pad_sequence(tars, batch_first=True, padding_value=-1)
    return padded_seqs, padded_tars, lengths

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# === DEBUG: Check batch structure and feature splitting ===
for x, y, lengths in loader:
    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    print("Lengths:", lengths)
    
    # Let's examine the actual feature structure
    print(f"\n=== Feature Structure Analysis ===")
    print(f"Total features in X: {x.shape[-1]}")
    
    # Current splitting logic (PROBLEMATIC)
    n_categorical = len(categorical_cols)
    n_coords = 2  # latitude, longitude
    n_weather = 4  # temperature, wind_speed, snowfall, rain
    n_other_numeric = 1  # volume
    
    print(f"Assuming: {n_categorical} categorical, {n_coords} coords, {n_weather} weather, {n_other_numeric} other")
    
    # Let's see what happens when we split
    cats = x[:, :, :n_categorical].long()
    coords = x[:, :, n_categorical:n_categorical + n_coords]
    weather_num = x[:, :, n_categorical + n_coords:n_categorical + n_coords + n_weather]
    other_numeric = x[:, :, n_categorical + n_coords + n_weather:]
    
    print(f"Categorical shape: {cats.shape}")
    print(f"Coords shape: {coords.shape}") 
    print(f"Weather shape: {weather_num.shape}")
    print(f"Other numeric shape: {other_numeric.shape}")
    
    # Check for NaN/Inf in the batch
    print(f"NaN in batch: {torch.isnan(x).any().item()}")
    print(f"Inf in batch: {torch.isinf(x).any().item()}")
    
    break


### MODEL ________________________________
import torch
import torch.nn as nn

class CoordEmbed(nn.Module):
    def __init__(self, input_dim=2, emb_size=8, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_size)
        )
    def forward(self, coords):   # coords: (N, 2) or (B*T, 2)
        return self.net(coords)

class WeatherEmbed(nn.Module):
    def __init__(self, input_dim=4, emb_size=8, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_size)
        )
    def forward(self, weather):  # weather: (N, 4)
        return self.net(weather)
class WasteLSTMModel(nn.Module):
    def __init__(self, cat_dims, coord_emb_size=8, weather_emb_size=8, other_numeric_size=1,
                 lstm_hidden=16, lstm_layers=1, padding_idx=0):
        super().__init__()
        self.n_cat = len(cat_dims)
        
        # Use smaller embedding dimensions
        emb_size = 8
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cd, embedding_dim=emb_size, padding_idx=padding_idx)
            for cd in cat_dims
        ])
        self.cat_emb_dim = emb_size * self.n_cat

        # Simpler embedding networks
        self.coord_emb = nn.Sequential(
            nn.Linear(2, coord_emb_size),
            nn.ReLU()
        )
        
        self.weather_emb = nn.Sequential(
            nn.Linear(4, weather_emb_size), 
            nn.ReLU()
        )

        self.lstm_input_size = self.cat_emb_dim + coord_emb_size + weather_emb_size + other_numeric_size
        
        print(f"LSTM input size: {self.lstm_input_size}")  # Debug
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=lstm_hidden,
            num_layers=lstm_layers, 
            batch_first=True
        )
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, coords, cats, weather_num, other_numeric, lengths=None):
        B, T, _ = cats.shape

        # Categorical embeddings
        cat_embs = [emb(cats[:, :, j].long()) for j, emb in enumerate(self.embeddings)]
        cat_emb = torch.cat(cat_embs, dim=2)

        # Coord embedding  
        coord_emb = self.coord_emb(coords)

        # Weather embedding
        weather_emb = self.weather_emb(weather_num)

        # Combine features
        x = torch.cat([cat_emb, coord_emb, weather_emb, other_numeric], dim=2)

        # Pack sequences if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        preds = self.head(out).squeeze(-1)
        return preds


## params_______________________________
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cat_dims = [len(le.classes_) for le in label_encoders.values()]

model = WasteLSTMModel(
    cat_dims=cat_dims,
    coord_emb_size=8,
    weather_emb_size=8, 
    other_numeric_size=1,
    lstm_hidden=8,
    lstm_layers=1
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Test forward pass first
print("\n=== Testing Forward Pass ===")
model.eval()
with torch.no_grad():
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        
        # CORRECT FEATURE SPLITTING
        n_cat = len(categorical_cols)
        cats = x[:, :, :n_cat].long()
        coords = x[:, :, n_cat:n_cat+2]  # latitude, longitude
        weather_num = x[:, :, n_cat+2:n_cat+6]  # temperature, wind, snowfall, rain  
        other_numeric = x[:, :, n_cat+6:n_cat+7]  # volume only
        
        print(f"Shapes - cats: {cats.shape}, coords: {coords.shape}, weather: {weather_num.shape}, other: {other_numeric.shape}")
        
        preds = model(coords, cats, weather_num, other_numeric, lengths)
        mask = y != -1
        
        print(f"Prediction range: {preds[mask].min().item():.4f} to {preds[mask].max().item():.4f}")
        print(f"Target range: {y[mask].min().item():.4f} to {y[mask].max().item():.4f}")
        
        loss = criterion(preds[mask], y[mask])
        print(f"Test loss: {loss.item():.4f}")
        break





from torch.utils.data import random_split

# ----------------------------
# 1️⃣ Split dataset for validation
# ----------------------------
val_frac = 0.1
val_size = int(len(dataset) * val_frac)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# ----------------------------
# 2️⃣ Training with gradient accumulation
# ----------------------------
accum_steps = 32  # accumulate gradients over 4 mini-batches
epochs = 5

print("\n=== Starting Training ===")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    optimizer.zero_grad()

    for batch_idx, (x, y, lengths) in enumerate(train_loader):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        
        # Split features
        n_cat = len(categorical_cols)
        cats = x[:, :, :n_cat].long()
        coords = x[:, :, n_cat:n_cat+2]
        weather_num = x[:, :, n_cat+2:n_cat+6] 
        other_numeric = x[:, :, n_cat+6:n_cat+7]
        
        preds = model(coords, cats, weather_num, other_numeric, lengths)
        
        mask = y != -1
        loss = criterion(preds[mask], y[mask])
        
        if torch.isnan(loss):
            continue
        
        # Normalize loss for accumulation
        (loss / accum_steps).backward()
        
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_train_loss = total_loss / batch_count if batch_count > 0 else float('nan')
    
    # ----------------------------
    # 3️⃣ Validation
    # ----------------------------
    model.eval()
    val_loss = 0
    val_count = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            
            n_cat = len(categorical_cols)
            cats = x[:, :, :n_cat].long()
            coords = x[:, :, n_cat:n_cat+2]
            weather_num = x[:, :, n_cat+2:n_cat+6] 
            other_numeric = x[:, :, n_cat+6:n_cat+7]
            
            preds = model(coords, cats, weather_num, other_numeric, lengths)
            
            # Mask padded positions
            mask = y != -1
            if mask.sum() == 0:  # skip if nothing to compute
                continue
            
            loss = criterion(preds[mask], y[mask])
            val_loss += loss.item()
            val_count += 1

    avg_val_loss = val_loss / val_count if val_count > 0 else float('nan')
