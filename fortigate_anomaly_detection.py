#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fortigate Network Traffic Anomaly Detection
-------------------------------------------
Questo script:
  1. Carica un file CSV Fortigate di traffico di rete
  2. Costruisce feature numeriche e codifica le variabili categoriche
  3. Esegue un Isolation Forest per rilevare anomalie
  4. Produce file di output con anomalie e spiegazioni

Output:
  - out/daily_features.csv
  - out/daily_iforest_scores.csv
  - out/daily_anomalies.csv
  - out/daily_anomaly_explanations.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# ============================================================
# CONFIGURAZIONE
# ============================================================
INPUT_FILE = "fortigate_part-20251021T144714Z_2.csv"
OUTPUT_DIR = "out"
N_ESTIMATORS = 200
CONTAMINATION = 0.01
RANDOM_STATE = 42

# ============================================================
# LETTURA FILE
# ============================================================
print(f"Caricamento file: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Shape originale: {df.shape}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# IDENTIFICAZIONE COLONNE TEMPORALI
# ============================================================
temporal_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "eventtime", "datetime", "uptime"])]
for tc in temporal_cols:
    try:
        df[tc + "_parsed"] = pd.to_datetime(df[tc], errors="coerce", utc=False, infer_datetime_format=True)
    except Exception:
        df[tc + "_parsed"] = pd.NaT

print(f"Colonne temporali trovate: {temporal_cols}")

# ============================================================
# COSTRUZIONE FEATURE NUMERICHE
# ============================================================
features = pd.DataFrame()

# Colonne numeriche native
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        features[col] = df[col]

# Conversione automatica di colonne "numeric-like"
for col in df.columns:
    if col not in features.columns:
        try:
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notnull().mean() > 0.05:
                features[col + "_num"] = conv
        except Exception:
            pass

# Feature temporali derivate
for col in [c for c in df.columns if "_parsed" in c]:
    features[col + "_hour"] = df[col].dt.hour
    features[col + "_weekday"] = df[col].dt.weekday

# Frequency encoding per alcune colonne categoriche note
cat_cols = [c for c in df.columns if any(x in c.lower() for x in ["srcip", "dstip", "srcserver", "transip", "country", "osname", "sessionid", "srcintf"])]
for col in cat_cols:
    features[col + "_enc"] = df[col].astype(str).map(df[col].astype(str).value_counts())

# Rimuovi colonne interamente NaN
features = features.dropna(axis=1, how="all").fillna(0)
print(f"Shape finale delle feature: {features.shape}")

# ============================================================
# SCALING
# ============================================================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(features)

# ============================================================
# ISOLATION FOREST
# ============================================================
clf = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE
)
clf.fit(X_scaled)
scores = clf.decision_function(X_scaled)
labels = clf.predict(X_scaled)

# -1 = anomalia, 1 = normale
is_anomaly = (labels == -1).astype(int)
df_scores = pd.DataFrame({
    "anomaly_score": scores,
    "is_anomaly": is_anomaly
})

# ============================================================
# IDENTIFICA ANOMALIE
# ============================================================
anomalies = df[is_anomaly == 1].copy()
print(f"Totale anomalie rilevate: {len(anomalies)}")

# ============================================================
# EXPLANATION: top 3 feature per z-score
# ============================================================
explanations = []
mean_vec = X_scaled.mean(axis=0)
std_vec = X_scaled.std(axis=0) + 1e-9
feature_names = features.columns

for idx in anomalies.index:
    z = np.abs((X_scaled[idx] - mean_vec) / std_vec)
    z = pd.Series(z, index=feature_names)
    top = z.nlargest(3)
    parts = [f"{feat} (z={z[feat]:.2f})" for feat in top.index]
    explanations.append("; ".join(parts))

anomalies["_explanation"] = explanations
anomalies["_anomaly_score"] = df_scores.loc[anomalies.index, "anomaly_score"].values

# ============================================================
# SALVATAGGIO FILE
# ============================================================
features.to_csv(os.path.join(OUTPUT_DIR, "daily_features.csv"), index=False)
df_scores.to_csv(os.path.join(OUTPUT_DIR, "daily_iforest_scores.csv"), index=False)
anomalies.to_csv(os.path.join(OUTPUT_DIR, "daily_anomalies.csv"), index=False)

explain_df = pd.DataFrame({
    "index": anomalies.index,
    "explanation": anomalies["_explanation"],
    "anomaly_score": anomalies["_anomaly_score"]
})
explain_df.to_csv(os.path.join(OUTPUT_DIR, "daily_anomaly_explanations.csv"), index=False)

print("File salvati in:", os.path.abspath(OUTPUT_DIR))
print("Completato ✅")
