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
import pyarrow.parquet as pq  # <-- needed for Parquet schema

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
    "sent_bytes": ["sent_bytes", "sentbytes", "bytes_sent", "out_bytes", "outbytes"],
    "rcvd_bytes": ["rcvd_bytes", "rcvdbytes", "bytes_rcvd", "in_bytes", "inbytes"],
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
}
ACTION_VALUES = ["accept", "deny", "client-rst", "server-rst", "timeout", "passthrough"]
ALIAS_LOWER_TO_CANON = {a.lower(): canon for canon, aliases in ALIASES.items() for a in aliases}

# ---------------------- Utilities: dates, IO ----------------------
def parse_folder_date(name: str) -> Optional[datetime]:
    # Prova a estrarre la data dalla stringa della cartella (tipo source-29.10.2025)
    m = re.match(r"^.+-(\d{2})\.(\d{2})\.(\d{4})$", name)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mth, d)
    except ValueError:
        return None

def list_source_day_folders(root: Path, start: Optional[str], end: Optional[str], target_source: Optional[str] = None) -> List[Tuple[str, datetime, Path]]:
    # Elenca le sottocartelle che rispettano il formato nome-data e filtra per intervallo temporale
    start_dt = dateparser.parse(start).date() if start else None
    print(start_dt)
    end_dt = dateparser.parse(end).date() if end else None
    print(end_dt)
    items: List[Tuple[str, datetime, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        dt = parse_folder_date(p.name)
        if not dt:
            continue
        src = p.name.split("-")[0]
        
        ## Applicazione filtro target_source
        # se è stato specificato un target_source e la cartella non lo rispetta bisogna saltarla
        if target_source and src != target_source: 
            continue
        # -----
        
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
def build_daily_features(df: pd.DataFrame, source: str, day: datetime) -> Dict[str, float]:
    if df.empty:
        return {
            "events_raw": 0.0,
            "events_after_day_filter": 0.0,
            "total_events": 0.0,
            "bytes_total": 0.0,
            "unique_src_ip": 0.0,
            "unique_dst_ip": 0.0,
            **{f"action_{a}_count": 0.0 for a in ACTION_VALUES},
            **{f"action_{a}_rate": 0.0 for a in ACTION_VALUES},
        }

    n_raw = float(len(df))
    df = canonicalize_columns(df)

    # Filter to folder day (if timestamp present)
    n_after = n_raw
    if "timestamp" in df.columns:
        ts = _normalize_timestamp_series(df["timestamp"])
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        mask = (ts >= day_start) & (ts < day_end)
        if mask.notna().any():
            df = df[mask.fillna(False)]
            n_after = float(len(df))

    total = float(len(df))

    # Bytes
    bytes_total = 0.0
    if "bytes" in df.columns:
        bytes_total = float(pd.to_numeric(df["bytes"], errors="coerce").fillna(0).sum())
    else:
        bt = 0.0
        if "sent_bytes" in df.columns:
            bt += float(pd.to_numeric(df["sent_bytes"], errors="coerce").fillna(0).sum())
        if "rcvd_bytes" in df.columns:
            bt += float(pd.to_numeric(df["rcvd_bytes"], errors="coerce").fillna(0).sum())
        bytes_total = bt

    # Cardinalità
    u_src = float(df["src_ip"].nunique()) if "src_ip" in df.columns else 0.0
    u_dst = float(df["dst_ip"].nunique()) if "dst_ip" in df.columns else 0.0

    # Actions
    if "action" in df.columns:
        actions_series = df["action"]
    elif "event_type" in df.columns:
        actions_series = df["event_type"]
    else:
        actions_series = pd.Series([], dtype=str)
    if isinstance(actions_series, pd.DataFrame):
        actions_series = actions_series.iloc[:, 0]
    actions = actions_series.astype(str).str.lower() if len(actions_series) else pd.Series([], dtype=str)

    feat = {
        "events_raw": n_raw,
        "events_after_day_filter": n_after,
        "total_events": total,
        "bytes_total": bytes_total,
        "bytes_per_event": (bytes_total / total) if total > 0 else 0.0,
        "unique_src_ip": u_src,
        "unique_dst_ip": u_dst,
    }

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

    for opt in ["proto", "service", "policy", "user", "host"]:
        if opt in df.columns:
            feat[f"unique_{opt}"] = float(df[opt].nunique())

    return feat

# ---------------------- Robust stats & explainability ----------------------
def robust_center_scale(values: pd.Series) -> Tuple[float, float]:
    med = float(values.median()); q75 = float(values.quantile(0.75)); q25 = float(values.quantile(0.25))
    iqr = max(q75 - q25, 1e-12)
    return med, iqr / 1.349  # ≈ sigma

def robust_deviation(x: float, med: float, iqr_scaled: float) -> float:
    if iqr_scaled <= 0: return 0.0
    return (x - med) / iqr_scaled

def training_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return per-feature (median, robust_sigma)."""
    stats: Dict[str, Tuple[float, float]] = {}
    numeric_cols = [c for c in train_df.columns if c not in {"source", "date"}]
    for c in numeric_cols:
        med, sig = robust_center_scale(pd.to_numeric(train_df[c], errors="coerce"))
        stats[c] = (med, sig)
    return stats

def score_and_explain(stats: Dict[str, Tuple[float, float]], today_feat: Dict[str, float]) -> Tuple[float, int, Dict[str, float], List[Dict[str, float]]]:
    """Return (score, voted_features, per_feature_z, explanations_list)."""
    per_feature_z: Dict[str, float] = {}
    explanations: List[Dict[str, float]] = []
    votes = 0
    voted_feats = 0
    for c, (med, sig) in stats.items():
        val = float(today_feat.get(c, 0.0))
        z = robust_deviation(val, med, sig)
        per_feature_z[c] = z
        vote = 0
        if abs(z) >= 3.5:
            vote = 2; votes += 2; voted_feats += 1
        elif abs(z) >= 2.5:
            vote = 1; votes += 1; voted_feats += 1
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
def iforest_score_from_training(train_df: pd.DataFrame, today_feat: Dict[str, float]) -> Optional[float]:
    if len(train_df) < 20:
        return None
    Xtr = train_df.drop(columns=[c for c in ["source","date"] if c in train_df.columns])
    xt = np.array([[today_feat.get(c, 0.0) for c in Xtr.columns]], dtype=np.float64)
    try:
        if _HAS_CUML and cuIsolationForest is not None:
            model = cuIsolationForest(max_samples="auto", n_estimators=256, random_state=42)
            gX = cudf.DataFrame.from_pandas(Xtr); model.fit(gX)
            s = float(model.decision_function(cudf.DataFrame(xt))[0])
        elif skIsolationForest is not None:
            model = skIsolationForest(max_samples="auto", n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(Xtr); s = float(model.decision_function(xt)[0])
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
        target_source: Optional[str] = None): # Nuovo argomento

    outdir.mkdir(parents=True, exist_ok=True) # Crea dir di output se non esiste

    items = list_source_day_folders(root, start, end, target_source=target_source) # Trova tutte le cartelle per le giornate (ora passiamo anche target_source)
    
    if not items:
        ## Messaggio di errore più specifico
        msg_src = f" for source '{target_source}'" if target_source else "" 
        print(f"No <source>-DD.MM.YYYY folders found in {root}{msg_src}"); return

    desired_lower = set(ALIAS_LOWER_TO_CANON.keys())

    # Build daily features
    daily_rows = []
    for (src, day, p) in tqdm(items, desc="Compute daily features"):
        df_day = gather_day_df(p, desired_lower)
        feat = build_daily_features(df_day, src, day)
        daily_rows.append({"source": src, "date": day.date().isoformat(), **feat})

    daily_df = pd.DataFrame(daily_rows).sort_values(["source","date"])
    (outdir / "daily_features.csv").write_text(daily_df.to_csv(index=False))

    # Train/Test
    scores = []
    if_rows = []
    explain_rows = []  # long-form explanations

    def summarize_top(explanations: List[Dict[str, float]], k: int) -> str:
        # pick features with a vote (z>=2.5) and sort by |z|
        interesting = [e for e in explanations if e["vote"] > 0]
        interesting.sort(key=lambda e: abs(e["z"]), reverse=True)
        pieces = []
        for e in interesting[:k]:
            sign = "+" if e["z"] >= 0 else "-"
            pieces.append(f"{e['feature']}:{sign}{abs(e['z']):.2f}({e['direction']})")
        return ", ".join(pieces)

    if holdout_days and holdout_days > 0:
        # Sliding N-5 train → last 5 test (or --holdout X)
        for src, group in daily_df.groupby("source"):
            group = group.reset_index(drop=True)
            N = len(group)
            if N <= holdout_days: continue
            W = N - holdout_days
            for i in range(W, N):
                train = group.iloc[i-W:i]
                today = group.iloc[i]
                stats = training_stats(train)
                score, vfeats, perz, expl = score_and_explain(stats, today.to_dict())
                scores.append({
                    "source": src, "date": today["date"], "score": score,
                    "voted_features": vfeats, "explain_top": summarize_top(expl, explain_top_k)
                })
                # store long-form explanations
                for e in expl:
                    explain_rows.append({
                        "source": src, "date": today["date"], **e
                    })
                if enable_iforest:
                    ifs = iforest_score_from_training(train, today.to_dict())
                    if ifs is not None:
                        if_rows.append({"source": src, "date": today["date"], "iforest_score": ifs})
    else:
        # Rolling mode
        for src, group in daily_df.groupby("source"):
            group = group.reset_index(drop=True)
            for i in range(len(group)):
                today = group.iloc[i]
                start_idx = max(0, i - window_days)
                train = group.iloc[start_idx:i]
                stats = training_stats(train)
                score, vfeats, perz, expl = score_and_explain(stats, today.to_dict())
                scores.append({
                    "source": src, "date": today["date"], "score": score,
                    "voted_features": vfeats, "explain_top": summarize_top(expl, explain_top_k)
                })
                for e in expl:
                    explain_rows.append({"source": src, "date": today["date"], **e})
                if enable_iforest:
                    ifs = iforest_score_from_training(train, today.to_dict())
                    if ifs is not None:
                        if_rows.append({"source": src, "date": today["date"], "iforest_score": ifs})

    # Assemble outputs
    scores_df = pd.DataFrame(scores)
    iforest_df = pd.DataFrame(if_rows) if if_rows else pd.DataFrame(columns=["source","date","iforest_score"])
    explanations_df = pd.DataFrame(explain_rows)
    if not explanations_df.empty:
        explanations_df.to_csv(outdir / "daily_anomaly_explanations.csv", index=False)

    # Merge context from features and label
    context_cols = [c for c in ["source","date","total_events","bytes_total","unique_src_ip","unique_dst_ip"] if c in daily_df.columns]
    features_view = daily_df[context_cols].copy() if context_cols else daily_df[["source","date"]].copy()
    result = scores_df.merge(features_view, on=["source","date"], how="left")
    if not iforest_df.empty:
        result = result.merge(iforest_df, on=["source","date"], how="left")

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
    anom = result[result["label"].isin(["MEDIUM","STRONG"])].copy()
    if anom.empty:
        print("(none flagged with current thresholds)")
    else:
        cols = ["source","date","label","score","voted_features","total_events","bytes_total","explain_top"]
        if "iforest_score" in anom.columns: cols.append("iforest_score")
        print(anom.sort_values(["score","voted_features"], ascending=False)[cols].head(print_limit).to_string(index=False))

    print("-", outdir / "daily_features.csv")
    print("-", outdir / "daily_anomalies.csv")
    if not iforest_df.empty:
        print("-", outdir / "daily_iforest_scores.csv")
    if not explanations_df.empty:
        print("-", outdir / "daily_anomaly_explanations.csv")

# devi lanciarlo con --iforest per farlo eseguire
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Daily per-source anomaly detection (Python, optional GPU).")
    ap.add_argument("root", type=str, help="Dataset root containing <source>-DD.MM.YYYY folders")
    
    ### NUOVO ARGOMENTO
    ap.add_argument("--source", type=str, default=None, help="Specifica il dispositivo da analizzare (es. watchguard)")
    
    ap.add_argument("--out", type=str, default="out", help="Output directory")
    ap.add_argument("--start", type=str, default=None, help="Start date (e.g. 2025-09-01)")
    ap.add_argument("--end", type=str, default=None, help="End date (e.g. 2025-10-23)")
    ap.add_argument("--window", type=int, default=35, help="Rolling window (used only if --holdout=0)")
    ap.add_argument("--holdout", type=int, default=5, help="Usa gli ultimi X giorni come test (N-X train). Per la tua richiesta: 5.")
    ap.add_argument("--iforest", action="store_true", help="Isolation Forest (GPU con cuML se disponibile)")
    # Labeling/printing/explainability controls
    ap.add_argument("--min-events", type=int, default=0, help="Minimo total_events per considerare un'anomalia (default 0)")
    ap.add_argument("--voted-features-min", type=int, default=2, help="Minimo di feature fuori soglia per MEDIUM (default 2)")
    ap.add_argument("--threshold-strong", type=float, default=1.5, help="Soglia STRONG sullo score (default 1.5)")
    ap.add_argument("--threshold-medium", type=float, default=1.0, help="Soglia MEDIUM sullo score (default 1.0)")
    ap.add_argument("--print-limit", type=int, default=50, help="Quante righe stampare a video (default 50)")
    ap.add_argument("--explain-top-k", type=int, default=5, help="Quante feature mostrare nel campo explain_top (default 5)")
    args = ap.parse_args()

    nomefile = sys.argv[0].split("/")[-1].replace(".py","")
    suffix_src = f"{args.source}" if args.source else "_ALL"
    args.out = args.out + "_" +nomefile + suffix_src + datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    params_file = utils.save_execution_params(args, Path(args.out))

    print("root:", args.root)
    print("source:", args.source)
    print("out:", args.out)
    print("start", args.start)
    print("end", args.end)
    print("window", args.window)
    print("iforest", args.iforest)
    print("holdout", args.holdout)
    print("min_events", args.min_events)
    print("voted_features_min", args.voted_features_min)
    print("threshold_strong", args.threshold_strong)
    print("threshold_medium", args.threshold_medium)
    print("print_limit", args.print_limit)
    print("explain_top_k", args.explain_top_k)

    # Passiamo args.source come target_source
    run(Path(args.root), Path(args.out), args.start, args.end, args.window, args.iforest, args.holdout,
        args.min_events, args.voted_features_min, args.threshold_strong, args.threshold_medium,
        args.print_limit, args.explain_top_k,
        target_source=args.source)

