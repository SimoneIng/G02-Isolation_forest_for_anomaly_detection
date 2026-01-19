import os
import re
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("./dataset")

def get_columns_from_file(path: Path):
    """Legge solo le colonne (header) del file senza caricare tutti i dati."""
    try:
        if path.name.lower().endswith(".csv") or path.name.lower().endswith(".csv.gz"):
            return pd.read_csv(path, nrows=0, sep=None, engine="python", compression="infer").columns.tolist()
    except Exception as e:
        return [f"Errore lettura: {str(e)}"]
    return []

def main():
    if not DATASET_DIR.exists():
        print(f"Errore: La cartella {DATASET_DIR} non esiste.")
        return

    # Teniamo traccia dei dispositivi già controllati per evitare duplicati
    seen_devices = set()
    
    folders = sorted([f for f in DATASET_DIR.iterdir() if f.is_dir()])
    
    print(f"{'DISPOSITIVO':<15} | {'COLONNE TROVATE'}")
    print("-" * 100)

    for folder in folders:
        # Estrae il nome dispositivo (es. da "fortigate-29.10.2025" -> "fortigate")
        match = re.match(r"^([a-zA-Z0-9_]+)-", folder.name)
        if not match:
            continue
            
        device_type = match.group(1)
        
        # Se abbiamo già analizzato un file di questo tipo, passiamo oltre
        if device_type in seen_devices:
            continue

        # Cerca un file valido nella cartella
        files = [p for p in folder.glob("*") if p.suffix.lower() in ['.csv', '.gz']]
        
        if files:
            first_file = files[0]
            columns = get_columns_from_file(first_file)
            print(f"{device_type:<15} | {columns}")
            seen_devices.add(device_type)
        else:
            print(f"{device_type:<15} | [Nessun file csv/parquet trovato]")

if __name__ == "__main__":
    main()