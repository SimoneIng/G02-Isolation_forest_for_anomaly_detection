#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import utils
import sys
import numpy as np
import pandas as pd
from dateutil import parser as dateparser
from tqdm import tqdm
import pyarrow.parquet as pq

# --- GPU (optional) ---
_GPU = False
_HAS_CUML = False
_HAS_CUDF = False
try:
    import cudf
    _HAS_CUDF = True
    _GPU = True
except Exception:
    cudf = None

try:
    from cuml.ensemble import IsolationForest as cuIsolationForest
    _HAS_CUML = True
    _GPU = True
except Exception:
    cuIsolationForest = None

try:
    from sklearn.ensemble import IsolationForest as skIsolationForest
except Exception:
    skIsolationForest = None

# ---------------------- CONFIG ----------------------
ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "@timestamp", "time", "eventtime", "log_time", "date", "datetime"],
    "sent_bytes": ["sent_bytes", "sentbytes", "bytes_sent", "out_bytes", "outbytes", "sentbyte"],
    "rcvd_bytes": ["rcvd_bytes", "rcvdbytes", "bytes_rcvd", "in_bytes", "inbytes", "rcvdbyte"],
    "bytes": ["bytes", "size"],
    "src_ip": ["src_ip", "srcip", "source_ip", "client_ip", "ip_src"],
    "dst_ip": ["dst_ip", "dstip", "destination_ip", "server_ip", "ip_dst"],
    "proto": ["proto", "protocol"],
    "service": ["service", "app", "application"],
    "policy": ["policyid", "policy", "rule", "rule_id"],
    "user": ["user", "username", "login", "account"],
    "host": ["host", "hostname", "computer", "device", "endpoint"],
    "action": ["action", "status", "decision", "result"],
    "event_type": ["event_type", "eventtype", "logid", "type", "subtype"],
    # --- NUOVO ALIAS PER DEVICE ID ---
    "device_id": ["devid", "device_id", "firebox_id", "hostname", "Hostname", "sn", "serial", "host", "firebox_name"],
}
ACTION_VALUES = ["accept", "deny", "client-rst", "server-rst", "timeout", "passthrough"]
ALIAS_LOWER_TO_CANON = {a.lower(): canon for canon, aliases in ALIASES.items() for a in aliases}

# ---------------------- Utilities ----------------------
def parse_folder_date(name: str) -> Optional[datetime]:
    m = re.match(r"^.+-(\d{2})\.(\d{2})\.(\d{4})$", name)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mth, d)
    except ValueError:
        return None

def list_source_day_folders(root: Path, start: Optional[str], end: Optional[str], target_source: Optional[str] = None) -> List[Tuple[str, datetime, Path]]:
    start_dt = dateparser.parse(start).date() if start else None
    end_dt = dateparser.parse(end).date() if end else None
    items: List[Tuple[str, datetime, Path]] = []
    
    if not root.exists():
        return []

    for p in root.iterdir():
        if not p.is_dir():
            continue
        dt = parse_folder_date(p.name)
        if not dt:
            continue
        
        src = p.name.split("-")[0]
        
        # Filtro per sorgente specifica (es. solo watchguard)
        if target_source and src != target_source:
            continue
            
        ddate = dt.date()
        if start_dt and ddate < start_dt: continue
        if end_dt and ddate > end_dt: continue
        items.append((src, dt, p))
        
    items.sort(key=lambda x: (x[0], x[1]))
    return items

def _normalize_timestamp_series(ts):
    try:
        s = pd.to_datetime(ts, errors="coerce", utc=True)
        return pd.Series(s.dt.tz_convert("UTC").dt.tz_localize(None))
    except Exception:
        return pd.Series(pd.NaT, index=getattr(ts, 'index', None))

def get_file_columns(path: Path) -> List[str]:
    low = path.name.lower()
    try:
        if low.endswith(".parquet"):
            pf = pq.ParquetFile(path)
            return [f.name for f in pf.schema_arrow]
        elif low.endswith(".csv") or low.endswith(".csv.gz"):
            df0 = pd.read_csv(path, nrows=0, sep=None, engine="python", compression="infer")
            return list(df0.columns)
        else:
            return []
    except Exception as e:
        print(f"[WARN] header detect failed for {path}: {e}")
        return []

def read_selective(path: Path, desired_lower: set) -> Optional[pd.DataFrame]:
    cols = get_file_columns(path)
    if not cols: return None
    want = [c for c in cols if c.lower() in desired_lower]
    if not want: return None
    low = path.name.lower()
    try:
        if low.endswith(".parquet"):
            return pd.read_parquet(path, columns=want)
        else:
            return pd.read_csv(path, usecols=want, sep=None, engine="python", compression="infer", on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] read_selective failed for {path}: {e}")
        return None

def gather_day_df(day_dir: Path, desired_lower: set) -> pd.DataFrame:
    frames = []
    files = [p for p in day_dir.rglob("*") if p.is_file() and re.search(r"\.(csv(\.gz)?|parquet)$", p.name, re.I)]
    for f in files:
        df_part = read_selective(f, desired_lower)
        if df_part is not None:
            frames.append(df_part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _coalesce_first_nonnull(cols: List[pd.Series]) -> pd.Series:
    s = cols[0].copy()
    for c in cols[1:]:
        s = s.where(~s.isna(), c)
    return s

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for canon, aliases in ALIASES.items():
        candidates = []
        if canon in df.columns: candidates.append(df[canon])
        for a in aliases:
            if a in df.columns and a != canon:
                candidates.append(df[a])
        if not candidates: continue
        df[canon] = _coalesce_first_nonnull(candidates)
        drop_cols = [a for a in aliases if a in df.columns and a != canon]
        if drop_cols: df = df.drop(columns=drop_cols)
    return df

# ---------------------- Feature Engineering ----------------------
def build_daily_features(df: pd.DataFrame, source: str, day: datetime) -> List[Dict[str, any]]:
    if df.empty: return []

    df = canonicalize_columns(df)
    
    # --- BLOCCO DEBUG ---
    if source == "fortigate":
        print(f"\n[DEBUG Fortigate] Righe caricate: {len(df)}")
        print(f"[DEBUG Fortigate] Colonne trovate dopo canonicalize: {df.columns.tolist()}")
        if "timestamp" in df.columns:
            print(f"[DEBUG Fortigate] Esempio timestamp: {df['timestamp'].iloc[0] if not df['timestamp'].empty else 'VUOTO'}")
    # --------------------

    # Filtro temporale interno
    # --- FIX PER TIMESTAMP SENZA DATA (Fortigate) ---
    if "timestamp" in df.columns:
        # Se il timestamp è solo ora (es. 12:08:12), aggiungiamo la data della cartella
        sample_ts = str(df["timestamp"].iloc[0])
        if len(sample_ts) <= 8 and ":" in sample_ts: # Rileva formato HH:MM:SS
            date_str = day.strftime("%Y-%m-%d")
            df["timestamp"] = date_str + " " + df["timestamp"].astype(str)
        
        ts = _normalize_timestamp_series(df["timestamp"])
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        mask = (ts >= day_start) & (ts < day_end)
        df = df[mask.fillna(False)]

    if df.empty: return []

    # Gestione Fallback Device ID (per pfsense)
    if "device_id" not in df.columns:
        df["device_id"] = f"{source}_default"
    else:
        df["device_id"] = df["device_id"].fillna(f"{source}_unknown").astype(str)

    ## IMPORTANTE: Raggruppamento per device_id
    results = []
    for dev_id, sub_df in df.groupby("device_id"):
        total = float(len(sub_df))
        
        # Bytes calculation
        bt = 0.0
        for c in ["bytes", "sent_bytes", "rcvd_bytes"]:
            if c in sub_df.columns:
                bt += float(pd.to_numeric(sub_df[c], errors="coerce").fillna(0).sum())

        ## Aggiungo device_id nella struttura dell'output. 
        feat = {
            "source": source,
            "device_id": dev_id,
            "date": day.date().isoformat(),
            "total_events": total,
            "bytes_total": bt,
            "bytes_per_event": (bt / total) if total > 0 else 0.0,
            "unique_src_ip": float(sub_df["src_ip"].nunique()) if "src_ip" in sub_df.columns else 0.0,
            "unique_dst_ip": float(sub_df["dst_ip"].nunique()) if "dst_ip" in sub_df.columns else 0.0,
        }

        # Actions
        act_col = "action" if "action" in sub_df.columns else ("event_type" if "event_type" in sub_df.columns else None)
        if act_col:
            actions = sub_df[act_col].astype(str).str.lower()
            vc = actions.value_counts()
            for a in ACTION_VALUES:
                feat[f"action_{a}_count"] = float(vc.get(a, 0.0))
                feat[f"action_{a}_rate"] = feat[f"action_{a}_count"] / total if total > 0 else 0.0
        
        for opt in ["proto", "service", "policy", "user"]:
            if opt in sub_df.columns:
                feat[f"unique_{opt}"] = float(sub_df[opt].nunique())

        results.append(feat)
    return results

# ---------------------- Stats & IF ----------------------
def robust_center_scale(values: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty: return 0.0, 1.0
    med = float(vals.median())
    iqr = max(float(vals.quantile(0.75) - vals.quantile(0.25)), 1e-12)
    return med, iqr / 1.349

def training_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats = {}
    cols = train_df.select_dtypes(include=[np.number]).columns
    for c in cols:
        stats[c] = robust_center_scale(train_df[c])
    return stats

def score_and_explain(stats: Dict[str, Tuple[float, float]], today_feat: Dict[str, any]):
    votes, voted_feats, expls = 0, 0, []
    for c, (med, sig) in stats.items():
        val = float(today_feat.get(c, 0.0))
        z = (val - med) / sig if sig > 0 else 0.0
        vote = 2 if abs(z) >= 3.5 else (1 if abs(z) >= 2.5 else 0)
        votes += vote
        if vote > 0: voted_feats += 1
        expls.append({"feature": c, "z": z, "vote": vote, "direction": "UP" if val >= med else "DOWN"})
    
    score = votes / max(1, voted_feats) if voted_feats > 0 else 0.0
    return score, voted_feats, expls

def iforest_score_from_training(train_df: pd.DataFrame, today_feat: Dict[str, any]) -> Optional[float]:
    if len(train_df) < 10: return None
    Xtr = train_df.select_dtypes(include=[np.number])
    xt = np.array([[today_feat.get(c, 0.0) for c in Xtr.columns]], dtype=np.float64)
    try:
        model = skIsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(Xtr)
        return float(model.decision_function(xt)[0])
    except: return None

# ---------------------- Runner ----------------------
def run(root: Path, outdir: Path, start: Optional[str], end: Optional[str],
        window_days: int, enable_iforest: bool, holdout_days: int,
        min_events: int, voted_features_min: int,
        threshold_strong: float, threshold_medium: float,
        print_limit: int, explain_top_k: int, target_source: Optional[str] = None):

    outdir.mkdir(parents=True, exist_ok=True)
    items = list_source_day_folders(root, start, end, target_source)
    if not items:
        print(f"No folders found for {target_source}"); return

    daily_rows = []
    desired_lower = set(ALIAS_LOWER_TO_CANON.keys())
    for (src, day, p) in tqdm(items, desc=f"Processing {target_source or 'all'}"):
        df_day = gather_day_df(p, desired_lower)
        feats = build_daily_features(df_day, src, day)
        daily_rows.extend(feats)

    if not daily_rows:
        print(f"[SKIP] Nessun dato valido estratto per {target_source}. Verificare date o nomi colonne.")
        return # Esce dalla funzione senza crashare lo script bash

    daily_df = pd.DataFrame(daily_rows)
    
    # Verifica che le colonne necessarie esistano davvero
    required_cols = ["source", "device_id", "date"]
    for col in required_cols:
        if col not in daily_df.columns:
            print(f"[ERROR] Colonna mancante nel dataset: {col}")
            return

    daily_df = daily_df.sort_values(required_cols)
    
    daily_df.to_csv(outdir / "daily_features.csv", index=False)

    scores, if_rows, explain_rows = [], [], []
    
    # Raggruppamento per Sorgente e per Device ID
    for (src, dev_id), group in daily_df.groupby(["source", "device_id"]):
        group = group.reset_index(drop=True)
        N = len(group)
        
        # Logica Holdout
        if holdout_days > 0 and N > holdout_days:
            W = N - holdout_days
            train = group.iloc[:W]
            stats = training_stats(train)
            for i in range(W, N):
                today = group.iloc[i]
                s, v, ex = score_and_explain(stats, today.to_dict())
                res = {"source": src, "device_id": dev_id, "date": today["date"], "score": s, "voted_features": v}
                scores.append(res)
                if enable_iforest:
                    ifs = iforest_score_from_training(train, today.to_dict())
                    if_rows.append({"source": src, "device_id": dev_id, "date": today["date"], "iforest_score": ifs})
        
    # Final Merge & Labeling
    if not scores: return
    res_df = pd.DataFrame(scores)
    if if_rows:
        res_df = res_df.merge(pd.DataFrame(if_rows), on=["source", "device_id", "date"], how="left")
    
    res_df["label"] = res_df.apply(lambda r: "STRONG" if r["score"] >= threshold_strong else ("MEDIUM" if r["score"] >= threshold_medium else "NORMAL"), axis=1)
    res_df.to_csv(outdir / "daily_anomalies.csv", index=False)
    print(f"\nAnalisi completata. Risultati in: {outdir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--iforest", action="store_true")
    ap.add_argument("--min-events", type=int, default=100)
    args = ap.parse_args()

    # Generazione cartella output specifica per source
    tag = args.source if args.source else "all"
    out_path = Path(f"{args.out}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    run(Path(args.root), out_path, args.start, args.end, 35, args.iforest, args.holdout,
        args.min_events, 2, 1.5, 1.0, 50, 5, target_source=args.source)