#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
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
    import cudf  # type: ignore
    _HAS_CUDF = True
    _GPU = True
except Exception:
    cudf = None

try:
    from cuml.ensemble import IsolationForest as cuIsolationForest  # type: ignore
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
    # --- Alias per Device ID ---
    "device_id": ["devid", "device_id", "firebox_id", "hostname", "Hostname", "sn", "serial", "host", "firebox_name"],
}
ACTION_VALUES = ["accept", "deny", "client-rst", "server-rst", "timeout", "passthrough"]
ALIAS_LOWER_TO_CANON = {a.lower(): canon for canon, aliases in ALIASES.items() for a in aliases}

# ---------------------- Utilities: dates, IO ----------------------
def parse_folder_date(name: str) -> Optional[datetime]:
    """Estrae la data dalla stringa della cartella (tipo source-29.10.2025)"""
    m = re.match(r"^.+-(\d{2})\.(\d{2})\.(\d{4})$", name)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mth, d)
    except ValueError:
        return None

def list_source_day_folders(root: Path, start: Optional[str], end: Optional[str], target_source: Optional[str] = None) -> List[Tuple[str, datetime, Path]]:
    """Elenca le sottocartelle che rispettano il formato nome-data e filtra per intervallo temporale"""
    start_dt = dateparser.parse(start).date() if start else None
    print(start_dt)
    end_dt = dateparser.parse(end).date() if end else None
    print(end_dt)
    
    items: List[Tuple[str, datetime, Path]] = []
    
    # Controllo esistenza directory
    if not root.exists():
        print(f"[ERROR] Directory {root} non esiste")
        return []
    
    for p in root.iterdir():
        if not p.is_dir():
            continue
        dt = parse_folder_date(p.name)
        if not dt:
            continue
        src = p.name.split("-")[0]
        
        # Filtro per sorgente specifica
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

# ---------------------- Header discovery + selective read ----------------------
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
    if not cols:
        return None
    want = [c for c in cols if c.lower() in desired_lower]
    if not want:
        return None
    low = path.name.lower()
    try:
        if low.endswith(".parquet"):
            if _HAS_CUDF:
                df = cudf.read_parquet(path, columns=want)
                return df.to_pandas()
            else:
                return pd.read_parquet(path, columns=want)
        else:
            if _HAS_CUDF and low.endswith((".csv", ".csv.gz")):
                df = cudf.read_csv(path, usecols=want, compression="infer")
                return df.to_pandas()
            else:
                return pd.read_csv(path, usecols=want, sep=None, engine="python",
                                   compression="infer", on_bad_lines="skip")
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

# ---------------------- Canonicalization / Dedup ----------------------
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
    dupes = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].unique()
    for name in dupes:
        same = [c for c in df.columns if c == name]
        if len(same) > 1:
            df[name] = _coalesce_first_nonnull([df[c] for c in same])
            df = df.drop(columns=same[1:])
    return df

# ---------------------- Feature Engineering ----------------------
def build_daily_features(df: pd.DataFrame, source: str, day: datetime, debug: bool = False) -> List[Dict[str, any]]:
    """
    Build features per source e per device_id.
    Ritorna una lista di dizionari, uno per ogni device_id trovato.
    """
    if df.empty:
        return []

    n_raw = float(len(df))
    df = canonicalize_columns(df)

    # --- Blocco DEBUG opzionale ---
    if debug and source == "fortigate":
        print(f"\n[DEBUG {source}] Righe caricate: {len(df)}")
        print(f"[DEBUG {source}] Colonne trovate: {df.columns.tolist()}")
        if "timestamp" in df.columns:
            print(f"[DEBUG {source}] Esempio timestamp: {df['timestamp'].iloc[0] if len(df) > 0 else 'VUOTO'}")

    # --- FIX per timestamp senza data (es. Fortigate con solo HH:MM:SS) ---
    n_after = n_raw
    if "timestamp" in df.columns:
        # Controlla se il timestamp è solo ora (es. 12:08:12)
        sample_ts = str(df["timestamp"].iloc[0]) if len(df) > 0 else ""
        if sample_ts and len(sample_ts) <= 8 and ":" in sample_ts:
            # Aggiungi la data della cartella
            date_str = day.strftime("%Y-%m-%d")
            df["timestamp"] = date_str + " " + df["timestamp"].astype(str)
        
        # Normalizza e filtra per il giorno della cartella
        ts = _normalize_timestamp_series(df["timestamp"])
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        mask = (ts >= day_start) & (ts < day_end)
        if mask.notna().any():
            df = df[mask.fillna(False)]
            n_after = float(len(df))

    if df.empty:
        return []

    # --- Gestione Device ID con fallback ---
    if "device_id" not in df.columns:
        df["device_id"] = f"{source}_default"
    else:
        df["device_id"] = df["device_id"].fillna(f"{source}_unknown").astype(str)

    # --- Raggruppa per device_id e calcola feature per ciascuno ---
    results = []
    for dev_id, sub_df in df.groupby("device_id"):
        total = float(len(sub_df))

        # Bytes
        bytes_total = 0.0
        if "bytes" in sub_df.columns:
            bytes_total = float(pd.to_numeric(sub_df["bytes"], errors="coerce").fillna(0).sum())
        else:
            bt = 0.0
            if "sent_bytes" in sub_df.columns:
                bt += float(pd.to_numeric(sub_df["sent_bytes"], errors="coerce").fillna(0).sum())
            if "rcvd_bytes" in sub_df.columns:
                bt += float(pd.to_numeric(sub_df["rcvd_bytes"], errors="coerce").fillna(0).sum())
            bytes_total = bt

        # Cardinalità IP
        u_src = float(sub_df["src_ip"].nunique()) if "src_ip" in sub_df.columns else 0.0
        u_dst = float(sub_df["dst_ip"].nunique()) if "dst_ip" in sub_df.columns else 0.0

        # Actions
        if "action" in sub_df.columns:
            actions_series = sub_df["action"]
        elif "event_type" in sub_df.columns:
            actions_series = sub_df["event_type"]
        else:
            actions_series = pd.Series([], dtype=str)
        
        if isinstance(actions_series, pd.DataFrame):
            actions_series = actions_series.iloc[:, 0]
        actions = actions_series.astype(str).str.lower() if len(actions_series) else pd.Series([], dtype=str)

        feat = {
            "source": source,
            "device_id": dev_id,
            "date": day.date().isoformat(),
            "events_raw": n_raw,  # Numero eventi prima del filtro temporale
            "events_after_day_filter": n_after,  # Dopo filtro temporale
            "total_events": total,  # Eventi per questo device_id
            "bytes_total": bytes_total,
            "bytes_per_event": (bytes_total / total) if total > 0 else 0.0,
            "unique_src_ip": u_src,
            "unique_dst_ip": u_dst,
        }

        # Conta e rate per azioni
        if total > 0 and len(actions) > 0:
            vc = actions.value_counts(dropna=False)
            for a in ACTION_VALUES:
                c = float(vc.get(a, 0.0))
                feat[f"action_{a}_count"] = c
                feat[f"action_{a}_rate"] = c / total
        else:
            for a in ACTION_VALUES:
                feat[f"action_{a}_count"] = 0.0
                feat[f"action_{a}_rate"] = 0.0

        # Cardinalità opzionali
        for opt in ["proto", "service", "policy", "user", "host"]:
            if opt in sub_df.columns:
                feat[f"unique_{opt}"] = float(sub_df[opt].nunique())

        results.append(feat)
    
    return results

# ---------------------- Robust stats & explainability ----------------------
def robust_center_scale(values: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return 0.0, 1.0
    med = float(vals.median())
    q75 = float(vals.quantile(0.75))
    q25 = float(vals.quantile(0.25))
    iqr = max(q75 - q25, 1e-12)
    return med, iqr / 1.349  # ≈ sigma

def robust_deviation(x: float, med: float, iqr_scaled: float) -> float:
    if iqr_scaled <= 0: return 0.0
    return (x - med) / iqr_scaled

def training_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return per-feature (median, robust_sigma)."""
    stats: Dict[str, Tuple[float, float]] = {}
    numeric_cols = [c for c in train_df.columns if c not in {"source", "device_id", "date"}]
    for c in numeric_cols:
        med, sig = robust_center_scale(train_df[c])
        stats[c] = (med, sig)
    return stats

def score_and_explain(stats: Dict[str, Tuple[float, float]], today_feat: Dict[str, any]) -> Tuple[float, int, Dict[str, float], List[Dict[str, any]]]:
    """Return (score, voted_features, per_feature_z, explanations_list)."""
    per_feature_z: Dict[str, float] = {}
    explanations: List[Dict[str, any]] = []
    votes = 0
    voted_feats = 0
    
    for c, (med, sig) in stats.items():
        val = float(today_feat.get(c, 0.0))
        z = robust_deviation(val, med, sig)
        per_feature_z[c] = z
        vote = 0
        if abs(z) >= 3.5:
            vote = 2
            votes += 2
            voted_feats += 1
        elif abs(z) >= 2.5:
            vote = 1
            votes += 1
            voted_feats += 1
        
        explanations.append({
            "feature": c,
            "value_today": val,
            "median": med,
            "robust_sigma": sig,
            "z": z,
            "vote": vote,
            "direction": "UP" if val >= med else "DOWN",
            "strength": "STRONG" if vote == 2 else ("MEDIUM" if vote == 1 else "OK"),
        })
    
    score = votes / max(1, voted_feats) if voted_feats > 0 else 0.0
    return float(score), int(voted_feats), per_feature_z, explanations

# ---------------------- Isolation Forest (optional) ----------------------
def iforest_score_from_training(train_df: pd.DataFrame, today_feat: Dict[str, any]) -> Optional[float]:
    if len(train_df) < 20:
        return None
    
    # Seleziona solo colonne numeriche escludendo metadati
    Xtr = train_df.drop(columns=[c for c in ["source", "device_id", "date"] if c in train_df.columns])
    xt = np.array([[today_feat.get(c, 0.0) for c in Xtr.columns]], dtype=np.float64)
    
    try:
        if _HAS_CUML and cuIsolationForest is not None:
            model = cuIsolationForest(max_samples="auto", n_estimators=256, random_state=42)
            gX = cudf.DataFrame.from_pandas(Xtr)
            model.fit(gX)
            s = float(model.decision_function(cudf.DataFrame(xt))[0])
        elif skIsolationForest is not None:
            model = skIsolationForest(max_samples="auto", n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(Xtr)
            s = float(model.decision_function(xt)[0])
        else:
            return None
        return s
    except Exception as e:
        print(f"[WARN] IF (train->1) failed: {e}")
        return None

# ---------------------- Runner ----------------------
def run(root: Path, outdir: Path, start: Optional[str], end: Optional[str],
        window_days: int, enable_iforest: bool, holdout_days: int,
        min_events: int, voted_features_min: int,
        threshold_strong: float, threshold_medium: float,
        print_limit: int, explain_top_k: int, 
        target_source: Optional[str] = None, debug: bool = False):

    outdir.mkdir(parents=True, exist_ok=True)

    items = list_source_day_folders(root, start, end, target_source=target_source)
    if not items:
        msg_src = f" for source '{target_source}'" if target_source else ""
        print(f"No <source>-DD.MM.YYYY folders found in {root}{msg_src}")
        return

    desired_lower = set(ALIAS_LOWER_TO_CANON.keys())

    # Build daily features (ora ritorna liste di dict, uno per device_id)
    daily_rows = []
    for (src, day, p) in tqdm(items, desc=f"Computing daily features for {target_source or 'all sources'}"):
        df_day = gather_day_df(p, desired_lower)
        feats = build_daily_features(df_day, src, day, debug=debug)
        daily_rows.extend(feats)  # extend perché ritorna una lista

    if not daily_rows:
        print(f"[SKIP] Nessun dato valido estratto per {target_source or 'all sources'}. Verificare date o nomi colonne.")
        return

    daily_df = pd.DataFrame(daily_rows)
    
    # Verifica colonne necessarie
    required_cols = ["source", "device_id", "date"]
    for col in required_cols:
        if col not in daily_df.columns:
            print(f"[ERROR] Colonna mancante nel dataset: {col}")
            return

    daily_df = daily_df.sort_values(required_cols)
    daily_df.to_csv(outdir / "daily_features.csv", index=False)

    # Train/Test
    scores = []
    if_rows = []
    explain_rows = []  # long-form explanations

    def summarize_top(explanations: List[Dict[str, any]], k: int) -> str:
        """Pick features with a vote (z>=2.5) and sort by |z|"""
        interesting = [e for e in explanations if e["vote"] > 0]
        interesting.sort(key=lambda e: abs(e["z"]), reverse=True)
        pieces = []
        for e in interesting[:k]:
            sign = "+" if e["z"] >= 0 else "-"
            pieces.append(f"{e['feature']}:{sign}{abs(e['z']):.2f}({e['direction']})")
        return ", ".join(pieces)

    if holdout_days and holdout_days > 0:
        # --- Modalità Holdout: ultimi X giorni come test ---
        for (src, dev_id), group in daily_df.groupby(["source", "device_id"]):
            group = group.reset_index(drop=True)
            N = len(group)
            if N <= holdout_days:
                continue
            W = N - holdout_days
            for i in range(W, N):
                train = group.iloc[i-W:i]
                today = group.iloc[i]
                stats = training_stats(train)
                score, vfeats, perz, expl = score_and_explain(stats, today.to_dict())
                scores.append({
                    "source": src,
                    "device_id": dev_id,
                    "date": today["date"],
                    "score": score,
                    "voted_features": vfeats,
                    "explain_top": summarize_top(expl, explain_top_k)
                })
                # store long-form explanations
                for e in expl:
                    explain_rows.append({
                        "source": src,
                        "device_id": dev_id,
                        "date": today["date"],
                        **e
                    })
                if enable_iforest:
                    ifs = iforest_score_from_training(train, today.to_dict())
                    if ifs is not None:
                        if_rows.append({
                            "source": src,
                            "device_id": dev_id,
                            "date": today["date"],
                            "iforest_score": ifs
                        })
    else:
        # --- Modalità Rolling Window ---
        for (src, dev_id), group in daily_df.groupby(["source", "device_id"]):
            group = group.reset_index(drop=True)
            for i in range(len(group)):
                today = group.iloc[i]
                start_idx = max(0, i - window_days)
                train = group.iloc[start_idx:i]
                
                if len(train) == 0:
                    continue
                
                stats = training_stats(train)
                score, vfeats, perz, expl = score_and_explain(stats, today.to_dict())
                scores.append({
                    "source": src,
                    "device_id": dev_id,
                    "date": today["date"],
                    "score": score,
                    "voted_features": vfeats,
                    "explain_top": summarize_top(expl, explain_top_k)
                })
                for e in expl:
                    explain_rows.append({
                        "source": src,
                        "device_id": dev_id,
                        "date": today["date"],
                        **e
                    })
                if enable_iforest:
                    ifs = iforest_score_from_training(train, today.to_dict())
                    if ifs is not None:
                        if_rows.append({
                            "source": src,
                            "device_id": dev_id,
                            "date": today["date"],
                            "iforest_score": ifs
                        })

    # Assemble outputs
    if not scores:
        print("[WARN] Nessuno score calcolato. Probabilmente non ci sono abbastanza dati per il training.")
        return

    scores_df = pd.DataFrame(scores)
    iforest_df = pd.DataFrame(if_rows) if if_rows else pd.DataFrame(columns=["source", "device_id", "date", "iforest_score"])
    explanations_df = pd.DataFrame(explain_rows)
    
    if not explanations_df.empty:
        explanations_df.to_csv(outdir / "daily_anomaly_explanations.csv", index=False)

    # Merge context from features and label
    context_cols = [c for c in ["source", "device_id", "date", "total_events", "bytes_total", "unique_src_ip", "unique_dst_ip"] 
                    if c in daily_df.columns]
    features_view = daily_df[context_cols].copy() if context_cols else daily_df[["source", "device_id", "date"]].copy()
    result = scores_df.merge(features_view, on=["source", "device_id", "date"], how="left")
    
    if not iforest_df.empty:
        result = result.merge(iforest_df, on=["source", "device_id", "date"], how="left")

    def _label_row(r):
        if pd.notna(r.get("total_events", np.nan)) and r.get("total_events", 0) < min_events:
            return "LOW_VOLUME"
        if r["score"] >= threshold_strong:
            return "STRONG"
        if (r["score"] >= threshold_medium) and (r.get("voted_features", 0) >= voted_features_min):
            return "MEDIUM"
        return "NORMAL"

    result["label"] = result.apply(_label_row, axis=1)

    # Save CSVs
    result.to_csv(outdir / "daily_anomalies.csv", index=False)
    if not iforest_df.empty:
        iforest_df.to_csv(outdir / "daily_iforest_scores.csv", index=False)

    # Print anomalies to stdout
    print("\n=== ANOMALIES FLAGGED ===")
    anom = result[result["label"].isin(["MEDIUM", "STRONG"])].copy()
    if anom.empty:
        print("(none flagged with current thresholds)")
    else:
        cols = ["source", "device_id", "date", "label", "score", "voted_features", "total_events", "bytes_total", "explain_top"]
        if "iforest_score" in anom.columns:
            cols.append("iforest_score")
        print(anom.sort_values(["score", "voted_features"], ascending=False)[cols].head(print_limit).to_string(index=False))

    print("\n=== Output files ===")
    print("-", outdir / "daily_features.csv")
    print("-", outdir / "daily_anomalies.csv")
    if not iforest_df.empty:
        print("-", outdir / "daily_iforest_scores.csv")
    if not explanations_df.empty:
        print("-", outdir / "daily_anomaly_explanations.csv")

# ---------------------- Main ----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Daily per-source and per-device anomaly detection (Python, optional GPU).")
    ap.add_argument("root", type=str, help="Dataset root containing <source>-DD.MM.YYYY folders")
    ap.add_argument("--source", type=str, default=None, help="Specifica il dispositivo da analizzare (es. watchguard)")
    ap.add_argument("--out", type=str, default="out", help="Output directory")
    ap.add_argument("--start", type=str, default=None, help="Start date (e.g. 2025-09-01)")
    ap.add_argument("--end", type=str, default=None, help="End date (e.g. 2025-10-23)")
    ap.add_argument("--window", type=int, default=35, help="Rolling window (used only if --holdout=0)")
    ap.add_argument("--holdout", type=int, default=5, help="Usa gli ultimi X giorni come test (N-X train). Default: 5.")
    ap.add_argument("--iforest", action="store_true", help="Isolation Forest (GPU con cuML se disponibile)")
    ap.add_argument("--min-events", type=int, default=0, help="Minimo total_events per considerare un'anomalia (default 0)")
    ap.add_argument("--voted-features-min", type=int, default=2, help="Minimo di feature fuori soglia per MEDIUM (default 2)")
    ap.add_argument("--threshold-strong", type=float, default=1.5, help="Soglia STRONG sullo score (default 1.5)")
    ap.add_argument("--threshold-medium", type=float, default=1.0, help="Soglia MEDIUM sullo score (default 1.0)")
    ap.add_argument("--print-limit", type=int, default=50, help="Quante righe stampare a video (default 50)")
    ap.add_argument("--explain-top-k", type=int, default=5, help="Quante feature mostrare nel campo explain_top (default 5)")
    ap.add_argument("--debug", action="store_true", help="Abilita output di debug per troubleshooting")
    args = ap.parse_args()

    nomefile = sys.argv[0].split("/")[-1].replace(".py", "")
    suffix_src = f"_{args.source}" if args.source else "_ALL"
    args.out = args.out + "_" + nomefile + suffix_src + datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    params_file = utils.save_execution_params(args, Path(args.out))

    print("=" * 60)
    print("root:", args.root)
    print("source:", args.source if args.source else "ALL")
    print("out:", args.out)
    print("start:", args.start)
    print("end:", args.end)
    print("window:", args.window)
    print("holdout:", args.holdout)
    print("iforest:", args.iforest)
    print("min_events:", args.min_events)
    print("voted_features_min:", args.voted_features_min)
    print("threshold_strong:", args.threshold_strong)
    print("threshold_medium:", args.threshold_medium)
    print("print_limit:", args.print_limit)
    print("explain_top_k:", args.explain_top_k)
    print("debug:", args.debug)
    print("GPU support:", "cuDF" if _HAS_CUDF else "No", "|", "cuML" if _HAS_CUML else "No")
    print("=" * 60)

    run(Path(args.root), Path(args.out), args.start, args.end, args.window, args.iforest, args.holdout,
        args.min_events, args.voted_features_min, args.threshold_strong, args.threshold_medium,
        args.print_limit, args.explain_top_k,
        target_source=args.source, debug=args.debug)