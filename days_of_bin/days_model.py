import pickle
import torch
import torch.nn as nn

############ MOOODEL________________________________________
class CoordEmbed(nn.Module):
    def __init__(self, input_dim=2, emb_size=8, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size)
        )
    def forward(self, coords):   # coords: (N, 2) or (B*T, 2)
        return self.net(coords)

class WeatherEmbed(nn.Module):
    def __init__(self, input_dim=4, emb_size=8, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size)
        )
    def forward(self, weather):  # weather: (N, 4)
        return self.net(weather)

class WasteLSTMModel(nn.Module):
    def __init__(self, cat_dims, coord_emb_size=8, weather_emb_size=8, other_numeric_size=1,
                 lstm_hidden=16, lstm_layers=1, padding_idx=0, dropout=0.1):
        super().__init__()
        self.n_cat = len(cat_dims)
        emb_size = 8

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cd, embedding_dim=emb_size, padding_idx=padding_idx)
            for cd in cat_dims
        ])
        self.cat_emb_dim = emb_size * self.n_cat

        # Coord & weather embeddings with dropout
        self.coord_emb = nn.Sequential(
            nn.Linear(2, coord_emb_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.weather_emb = nn.Sequential(
            nn.Linear(4, weather_emb_size), 
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm_input_size = self.cat_emb_dim + coord_emb_size + weather_emb_size + other_numeric_size
        
        # LayerNorm before LSTM
        self.feature_norm = nn.LayerNorm(self.lstm_input_size)
        
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

        # Coord & weather embeddings
        coord_emb = self.coord_emb(coords)
        weather_emb = self.weather_emb(weather_num)

        # Combine features
        x = torch.cat([cat_emb, coord_emb, weather_emb, other_numeric], dim=2)
        x = self.feature_norm(x)

        # Pack sequences if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        preds = self.head(out).squeeze(-1)
        return preds

############ LOGSSSSSSSSSSSSSSSSS_________________________________
class WasteLSTMPredictor:
    def __init__(self, artifacts_path="waste_lstm_artifacts.pkl", device='cpu'):
        self.device = device
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)

        self.label_encoders = artifacts["label_encoders"]
        self.scaler = artifacts["scaler"]
        self.categorical_cols = artifacts["categorical_cols"]
        self.numeric_cols = artifacts["numeric_cols"]
        self.cat_dims = artifacts["cat_dims"]
        self.model_params = artifacts["model_params"]

        # Load model
        self.model = WasteLSTMModel(cat_dims=self.cat_dims, **self.model_params)
        self.model.load_state_dict(artifacts["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, df):
        # Encode categorical
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

        # Scale numeric
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df

    def predict(self, df):
        df = self.preprocess(df)
        cats = torch.tensor(df[self.categorical_cols].values, dtype=torch.long, device=self.device).unsqueeze(1)
        coords = torch.tensor(df[["Konteinerio_latitude","Konteinerio_longitude"]].values, dtype=torch.float32, device=self.device).unsqueeze(1)
        weather_num = torch.tensor(df[["temperature_2m_mean (В°C)","wind_speed_10m_max (km/h)","snowfall_sum (cm)","rain_sum (mm)"]].values, dtype=torch.float32, device=self.device).unsqueeze(1)
        other_numeric = torch.tensor(df[["Konteinerio_volume"]].values, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            preds = self.model(coords, cats, weather_num, other_numeric)
            pred_days = torch.expm1(preds)
        return pred_days.squeeze().cpu().numpy()