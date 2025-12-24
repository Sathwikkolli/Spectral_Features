import os
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, DetCurveDisplay
import matplotlib.pyplot as plt

import umap
import plotly.express as px

from spafe.features.mfcc import mfcc  # ✅ spafe MFCC


# ======================================================
# REPRODUCIBILITY
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ======================================================
# PATHS (protocol in spectral/, code in spectral/src/)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../spectral/src
PROJECT_DIR = os.path.dirname(BASE_DIR)                   # .../spectral

PROTOCOL_CSV  = os.path.join(PROJECT_DIR, "oc_protocol_eval1000.csv")
PROTOCOL_ROOT = "/data/FF_V2/FF_V2"
REAL_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures"
SPEAKER       = "Donald_Trump"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(BASE_DIR, "results/mfcc")
MODEL_DIR = os.path.join(BASE_DIR, "results/models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ======================================================
# MFCC (FROZEN) — spafe, 20 coeffs, include energy, no deltas, mean pooling
# ======================================================
SR = 16000
DURATION = 10.0
N_FFT = 512
N_FILTERS = 60
N_MFCC = 20
INCLUDE_ENERGY = True   # ✅ you said "yes"


# ======================================================
# DeepSVDD (FROZEN) — same as LFCC
# ======================================================
SVDD_NU = 0.05
SVDD_EPOCHS = 30
SVDD_LR = 1e-3
SVDD_WEIGHT_DECAY = 1e-4

H1 = 64
H2 = 32
NEG_SLOPE = 0.1


class DeepSVDD(nn.Module):
    def __init__(self, input_dim: int):
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
# HELPERS
# ======================================================
def resolve_path(p: str) -> str:
    # FF protocol path mapping to real data root (same pattern as your LFCC)
    return p.replace(PROTOCOL_ROOT, REAL_ROOT).replace("/Original/", "/-/")

def label_from_path(p: str) -> str:
    return "Real" if "/Original/" in p else "Fake"

def attack_from_path(p: str) -> str:
    if "/Original/" in p:
        return "Real"
    token = f"/{SPEAKER}/"
    return p.split(token, 1)[1].split("/", 1)[0]


# ======================================================
# MFCC EXTRACTION (spafe)
# ======================================================
def extract_mfcc_vector(p: str) -> np.ndarray:
    y, _ = librosa.load(resolve_path(p), sr=SR, duration=DURATION)

    feats = mfcc(
        sig=y,
        fs=SR,
        num_ceps=N_MFCC,
        nfilts=N_FILTERS,
        nfft=N_FFT,
        use_energy=INCLUDE_ENERGY
    )
    # mean pooling (same as LFCC)
    return np.mean(feats, axis=0).astype(np.float32)

def build_matrix(paths) -> np.ndarray:
    return np.vstack([extract_mfcc_vector(p) for p in tqdm(paths, desc="MFCC")])


# ======================================================
# METRICS
# ======================================================
def compute_eer(sr: np.ndarray, sf: np.ndarray):
    y = np.concatenate([np.zeros_like(sr), np.ones_like(sf)])
    s = np.concatenate([sr, sf])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx]), float(thr[idx]), fpr, tpr


# ======================================================
# TRAINING (SOFT-BOUNDARY)
# ======================================================
def train_deepsvdd(X_train: np.ndarray):
    model = DeepSVDD(X_train.shape[1]).to(DEVICE)
    X = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        center = model(X).mean(0)
        dist = torch.sum((model(X) - center) ** 2, dim=1)
        R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    opt = optim.Adam(model.parameters(), lr=SVDD_LR, weight_decay=SVDD_WEIGHT_DECAY)

    model.train()
    for _ in range(SVDD_EPOCHS):
        opt.zero_grad()
        z = model(X)
        dist = torch.sum((z - center) ** 2, dim=1)
        loss = R**2 + (1 / SVDD_NU) * torch.mean(torch.clamp(dist - R**2, min=0))
        loss.backward()
        opt.step()

        with torch.no_grad():
            dist = torch.sum((model(X) - center) ** 2, dim=1)
            R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    tau = float(R.item() ** 2)
    model.eval()
    return model, center, tau


# ======================================================
# MAIN
# ======================================================
def main():
    df = pd.read_csv(PROTOCOL_CSV)
    df["label"] = df.audiofilepath.apply(label_from_path)
    df["attack"] = df.audiofilepath.apply(attack_from_path)

    real_train = df[(df.label == "Real") & (df.split == "train")]
    eval_df = df[df.split == "eval"]
    real_eval = eval_df[eval_df.label == "Real"]
    fake_eval = eval_df[eval_df.label == "Fake"]

    # ---------- FEATURES ----------
    Xtr = build_matrix(real_train.audiofilepath.values)
    Xr  = build_matrix(real_eval.audiofilepath.values)
    Xf  = build_matrix(fake_eval.audiofilepath.values)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xr_s  = scaler.transform(Xr)
    Xf_s  = scaler.transform(Xf)

    # ---------- TRAIN ----------
    model, center, tau = train_deepsvdd(Xtr_s)

    # ---------- SCORES ----------
    with torch.no_grad():
        Xr_t = torch.tensor(Xr_s, dtype=torch.float32).to(DEVICE)
        Xf_t = torch.tensor(Xf_s, dtype=torch.float32).to(DEVICE)

        Zr = model(Xr_t)
        Zf = model(Xf_t)

        sr = torch.sum((Zr - center) ** 2, dim=1).cpu().numpy()
        sf = torch.sum((Zf - center) ** 2, dim=1).cpu().numpy()

    FRR = float(np.mean(sr > tau))
    FAR = float(np.mean(sf <= tau))
    eer, eer_thr, fpr, tpr = compute_eer(sr, sf)
    auc = float(roc_auc_score(
        np.concatenate([np.zeros_like(sr), np.ones_like(sf)]),
        np.concatenate([sr, sf])
    ))

    print("\n=== MFCC + DeepSVDD ===")
    print(f"Dim: {Xtr_s.shape[1]}")
    print(f"FRR: {FRR:.4f} | FAR: {FAR:.4f}")
    print(f"EER: {eer:.4f} | AUC: {auc:.4f}")
    print(f"tau: {tau:.6f}")

    # ---------- SAVE MODEL ----------
    model_path = os.path.join(MODEL_DIR, "mfcc_deepsvdd_trump.pt")
    torch.save({
        "net_state_dict": model.state_dict(),
        "center": center.detach().cpu(),
        "radius": torch.tensor(np.sqrt(tau)),   # stored as tensor; load with float(radius)**2
        "scaler": scaler,
        "mfcc_params": {
            "sr": SR, "duration": DURATION, "n_fft": N_FFT, "n_filters": N_FILTERS,
            "n_mfcc": N_MFCC, "use_energy": INCLUDE_ENERGY, "pool": "mean"
        },
        "svdd_params": {
            "nu": SVDD_NU, "epochs": SVDD_EPOCHS, "lr": SVDD_LR, "weight_decay": SVDD_WEIGHT_DECAY,
            "h1": H1, "h2": H2, "neg_slope": NEG_SLOPE
        }
    }, model_path)
    print(f"[SAVED] MFCC DeepSVDD → {model_path}")

    # ======================================================
    # VISUALIZATIONS (same structure as LFCC)
    # ======================================================

    # 1) Score distribution
    plt.figure()
    plt.hist(sr, bins=50, alpha=0.6, label="Real")
    plt.hist(sf, bins=50, alpha=0.6, label="Fake")
    plt.axvline(tau, color="red", linestyle="--", label="τ")
    plt.legend()
    plt.title("MFCC – SVDD Score Distribution")
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"))
    plt.close()

    # 2) ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC – MFCC + DeepSVDD")
    plt.savefig(os.path.join(OUT_DIR, "roc.png"))
    plt.close()

    # 3) DET
    DetCurveDisplay(fpr=fpr, fnr=1 - tpr).plot()
    plt.title("DET – MFCC + DeepSVDD")
    plt.savefig(os.path.join(OUT_DIR, "det.png"))
    plt.close()

    # 4) Attack-wise FAR
    # IMPORTANT: sf is aligned with fake_eval order (we built Xf from fake_eval.audiofilepath.values)
    rows = []
    fake_attacks = fake_eval.attack.values
    uniq_attacks = sorted(np.unique(fake_attacks))

    for atk in uniq_attacks:
        idx = (fake_attacks == atk)
        far_atk = float(np.mean(sf[idx] <= tau))
        rows.append((atk, far_atk, int(np.sum(idx))))

    atk_df = pd.DataFrame(rows, columns=["attack", "FAR", "count"])
    atk_df.to_csv(os.path.join(OUT_DIR, "attack_wise_far.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(atk_df["attack"], atk_df["FAR"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Attack-wise FAR – MFCC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attack_wise_far.png"))
    plt.close()

    # 5) UMAP – LATENT SPACE (Plotly)
    reducer_lat = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    Z_lat = reducer_lat.fit_transform(np.vstack([Zr.cpu().numpy(), Zf.cpu().numpy()]))

    labels_lat = ["Real"] * len(Zr) + list(fake_attacks)
    df_lat = pd.DataFrame({"x": Z_lat[:, 0], "y": Z_lat[:, 1], "label": labels_lat})

    px.scatter(
        df_lat, x="x", y="y", color="label",
        title="UMAP – DeepSVDD Latent Space (MFCC)"
    ).write_html(os.path.join(OUT_DIR, "umap_latent.html"))

    # 6) UMAP – RAW MFCC (Plotly)
    reducer_raw = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    Z_raw = reducer_raw.fit_transform(np.vstack([Xr_s, Xf_s]))

    labels_raw = ["Real"] * len(Xr_s) + list(fake_attacks)
    df_raw = pd.DataFrame({"x": Z_raw[:, 0], "y": Z_raw[:, 1], "label": labels_raw})

    px.scatter(
        df_raw, x="x", y="y", color="label",
        title="UMAP – Raw MFCC Features"
    ).write_html(os.path.join(OUT_DIR, "umap_raw.html"))

    print("\nSaved all MFCC outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
