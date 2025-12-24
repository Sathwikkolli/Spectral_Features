import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

import umap
import plotly.express as px


# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# PATHS
# ======================================================
DS_WILD_ROOT = Path("/nfs/turbo/umd-hafiz/issf_server_data/ds_wild")
PROTOCOL_CSV = DS_WILD_ROOT / "protocols/meta.csv"
AUDIO_ROOT   = DS_WILD_ROOT / "release_in_the_wild"

MODEL_PATH = "/home/ksathwik/projects/spectral/src/results/models/stm_deepsvdd_trump.pt"

OUT_DIR = "results/stm_itw_test_only"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# STM CONFIG (MUST MATCH TRAINING)
# ======================================================
SR = 16000
DURATION = 10.0

N_FFT = 512
WIN_LENGTH = int(0.025 * SR)
HOP_LENGTH = int(0.010 * SR)
N_MELS = 128
FMIN = 20
FMAX = 7600

TEMP_BANDS_HZ = [(0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0)]
SPEC_BANDS_CPM = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40)]

STM_DIM = len(TEMP_BANDS_HZ) * len(SPEC_BANDS_CPM)


# ======================================================
# DeepSVDD MODEL (EXACT SAME AS TRAINING)
# ======================================================
H1 = 64
H2 = 32
NEG_SLOPE = 0.1


class DeepSVDD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, H1, bias=False),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Linear(H1, H2, bias=False),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Linear(H2, input_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# STM EXTRACTION (Option A)
# ======================================================
def _logmel(y):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    return np.log1p(S).astype(np.float32)

def extract_stm(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    M = _logmel(y)
    M = M - np.mean(M)

    P = np.abs(np.fft.fftshift(np.fft.fft2(M))) ** 2
    F, T = P.shape

    spec_freq = np.abs(np.fft.fftshift(np.fft.fftfreq(F, d=1.0)))
    temp_freq = np.abs(np.fft.fftshift(np.fft.fftfreq(T, d=HOP_LENGTH / SR)))

    feats = []
    for (s_lo, s_hi) in SPEC_BANDS_CPM:
        s_mask = (spec_freq >= s_lo) & (spec_freq < s_hi)
        for (t_lo, t_hi) in TEMP_BANDS_HZ:
            t_mask = (temp_freq >= t_lo) & (temp_freq < t_hi)
            if not np.any(s_mask) or not np.any(t_mask):
                feats.append(0.0)
            else:
                feats.append(float(P[np.ix_(s_mask, t_mask)].mean()))

    return np.log1p(np.array(feats, dtype=np.float32))


# ======================================================
# STEP 1: LOAD ds_wild PROTOCOL (DONALD TRUMP ONLY)
# ======================================================
print("\n[STEP 1] Loading ds_wild protocol")

df = pd.read_csv(PROTOCOL_CSV)
df["label"] = df["label"].str.strip().str.lower()
df_trump = df[df["speaker"] == "Donald Trump"]

print(f"[PROTO] Total entries        : {len(df)}")
print(f"[PROTO] Donald Trump entries : {len(df_trump)}")

assert len(df_trump) > 0, "No Donald Trump samples found"


# ======================================================
# STEP 2: EXTRACT STM EMBEDDINGS
# ======================================================
print("\n[STEP 2] Extracting STM (ITW)")

X, y = [], []

for _, row in tqdm(df_trump.iterrows(), total=len(df_trump)):
    wav_path = AUDIO_ROOT / row["file"]
    if not wav_path.exists():
        continue
    try:
        X.append(extract_stm(str(wav_path)))
        y.append(0 if row["label"] == "bona-fide" else 1)
    except Exception as e:
        print(f"[SKIP] {wav_path} | {e}")

X = np.vstack(X)
y = np.array(y)

print(f"[STM] Shape      : {X.shape}")
print(f"[STM] Bona-fide  : {(y==0).sum()}")
print(f"[STM] Spoof      : {(y==1).sum()}")


# ======================================================
# STEP 3: LOAD STM DeepSVDD MODEL
# ======================================================
print("\n[STEP 3] Loading STM DeepSVDD")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

scaler: StandardScaler = ckpt["scaler"]

model = DeepSVDD(STM_DIM).to(DEVICE)
model.load_state_dict(ckpt["net_state_dict"])
model.eval()

center = ckpt["center"].to(DEVICE)
radius = ckpt["radius"]
tau = float(radius) ** 2

print("[MODEL] Loaded successfully")


# ======================================================
# STEP 4: COMPUTE SVDD SCORES
# ======================================================
print("\n[STEP 4] Computing SVDD scores")

X_s = scaler.transform(X)

with torch.no_grad():
    Xt = torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
    Z = model(Xt)
    scores = torch.sum((Z - center) ** 2, dim=1).cpu().numpy()


# ======================================================
# STEP 5: METRICS
# ======================================================
real_scores = scores[y == 0]
spoof_scores = scores[y == 1]

FRR = np.mean(real_scores > tau)

y_full = np.concatenate([np.zeros_like(real_scores), np.ones_like(spoof_scores)])
s_full = np.concatenate([real_scores, spoof_scores])

fpr, tpr, _ = roc_curve(y_full, s_full)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

print("\n========== STM ITW (ds_wild) ==========")
print(f"FRR (real) : {FRR:.6f}")
print(f"EER        : {eer:.6f}")
print(f"τ (SVDD)   : {tau:.6f}")
print("======================================")


# ======================================================
# STEP 6: SCORE DISTRIBUTION
# ======================================================
plt.figure()
plt.hist(real_scores, bins=50, alpha=0.6, label="Bona-fide")
plt.hist(spoof_scores, bins=50, alpha=0.6, label="Spoof")
plt.axvline(tau, color="red", linestyle="--", label="τ")
plt.legend()
plt.title("STM + DeepSVDD — ITW (Donald Trump)")
plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"))
plt.close()


# ======================================================
# STEP 7A: UMAP – RAW STM
# ======================================================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
Z_raw = reducer.fit_transform(X_s)

df_raw = pd.DataFrame({
    "x": Z_raw[:, 0],
    "y": Z_raw[:, 1],
    "label": ["Bona-fide" if l == 0 else "Spoof" for l in y]
})

px.scatter(
    df_raw, x="x", y="y", color="label",
    title="UMAP – RAW STM (ds_wild, Donald Trump)",
    opacity=0.7
).write_html(os.path.join(OUT_DIR, "umap_raw_stm.html"))


# ======================================================
# STEP 7B: UMAP – LATENT DeepSVDD
# ======================================================
with torch.no_grad():
    Z_lat = model(
        torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
    ).cpu().numpy()

Z_lat_umap = reducer.fit_transform(Z_lat)

df_lat = pd.DataFrame({
    "x": Z_lat_umap[:, 0],
    "y": Z_lat_umap[:, 1],
    "label": ["Bona-fide" if l == 0 else "Spoof" for l in y]
})

px.scatter(
    df_lat, x="x", y="y", color="label",
    title="UMAP – DeepSVDD Latent Space (STM, ds_wild)",
    opacity=0.7
).write_html(os.path.join(OUT_DIR, "umap_latent_svdd.html"))


print(f"\n[DONE] STM ITW results saved to: {OUT_DIR}")
