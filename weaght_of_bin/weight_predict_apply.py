import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd

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

class GetWeight:
    def __init__(self, artifacts_path="artifacts.pkl", model_path="weight_model.pth"):
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        self.label_encoders = artifacts["label_encoders"]
        self.scaler = artifacts["scaler"]
        self.target_scaler = artifacts["target_scaler"]
        self.coord_features = artifacts["coord_features"]
        self.numeric_features = artifacts["numeric_features"]
        self.categorical_features = artifacts["categorical_features"]

        cat_dims = [len(self.label_encoders[feat].classes_) for feat in self.categorical_features]
        model = WasteWeightModel(num_numeric=len(self.numeric_features),
                                 cat_dims=cat_dims,
                                 emb_size=64,
                                 hidden_dim=64,
                                 hid_for_pos=32,
                                 depth=2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        self.model = model

    def preprocess_input(self, input_dict):
        # Numeric + coordinates
        values = [input_dict[feat] for feat in self.numeric_features + self.coord_features]
        df_input = pd.DataFrame([values], columns=self.numeric_features + self.coord_features)
        scaled = self.scaler.transform(df_input)
        numeric_scaled = torch.tensor(scaled[:, :len(self.numeric_features)], dtype=torch.float32)
        coords_scaled = torch.tensor(scaled[:, len(self.numeric_features):], dtype=torch.float32)

        # Categorical
        cat_ids = []
        for feat in self.categorical_features:
            le = self.label_encoders[feat]
            val = input_dict[feat]
            if val not in le.classes_:
                raise ValueError(f"Unknown category '{val}' for feature '{feat}'")
            cat_id = le.transform([val])[0]
            cat_ids.append(cat_id)
        cats = torch.tensor([cat_ids], dtype=torch.long)

        return coords_scaled, numeric_scaled, cats

    def predict(self, input_dict):
        self.model.eval()
        coords_scaled, numeric_scaled, cats = self.preprocess_input(input_dict)
        with torch.no_grad():
            pred_scaled = self.model(coords_scaled, numeric_scaled, cats)
        pred_real = self.target_scaler.inverse_transform(pred_scaled.numpy())
        return float(np.maximum(0, pred_real[0,0]))
