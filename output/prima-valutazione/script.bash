#!/bin/bash

# ANALISI DELLE ANOMALIE PER OGNI DISPOSITIVO
DEVICES=("fortigate" "mac_windows" "pfsense" "vsphere_vm" "watchguard")

DATASET_DIR="./dataset"

for DEV in "${DEVICES[@]}"; do
    echo "Avvio analisi per $DEV"

    Python3 anomaly-detection_v1.py "$DATASET_DIR" \
      --source "$DEV" \
      --iforest \
      --start "2025-09-01" \
      --end "2025-10-23" \
      --holdout 5 \
      --out "out_$DEV" \
      --min-events 100

    echo "Analisi $DEV completata"
done 