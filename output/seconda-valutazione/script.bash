#!/bin/bash

# --- CONFIGURAZIONE ---
# Lista di tutti i dispositivi da processare
DEVICES=("fortigate" "mac_windows" "pfsense" "vsphere_vm" "watchguard")

DATASET_DIR="./dataset"
OUTPUT_BASE_DIR="./output/seconda-valutazione"
PYTHON_SCRIPT="anomaly-detection_v2.py"

echo "INIZIO ANALISI ANOMALIE DI RETE (Per Device ID)"

for DEV in "${DEVICES[@]}"; do
    echo ""
    echo ">>> Elaborazione dispositivo: $DEV"
    
    # Definiamo una sottocartella specifica per il dispositivo dentro i risultati
    # Esempio: ./results_analysis/fortigate
    DEV_OUT_DIR="$OUTPUT_BASE_DIR/$DEV"

    # Esecuzione dello script Python
    # Passiamo --source per filtrare le cartelle nel dataset
    # Passiamo --out come prefisso per la cartella di output finale
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
      --source "$DEV" \
      --iforest \
      --start "2025-09-01" \
      --end "2025-10-23" \
      --holdout 5 \
      --out "$DEV_OUT_DIR" \
      --min-events 100

    if [ $? -eq 0 ]; then
        echo ">>> Analisi per $DEV completata con successo."
    else
        echo "!!! Errore durante l'analisi di $DEV."
    fi
    
    echo "---------------------------------------------------------"
done

echo "ANALISI COMPLETATA"
echo "I risultati sono disponibili in: $OUTPUT_BASE_DIR"
