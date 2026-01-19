# Obiettivo

Aumentare la precisione dell'algoritmo passando da un'analisi per tipo di sorgente ad un'analisi per singolo dispositivo. 

Motivazione: Evitare che un dispositivo ad alto traffico nasconda anomalie di un dispositivo a basso traffico dello stesso produttore. 

## Modifiche apportate

1. Identificazione Univoca (Device ID): 

    - Il dizionario `ALIASES` è stato aggiornato per includere colonne identificative che variano in base al dispositivo. 
    - Implementata logica di fallback se non viene trovato nessun campo ID (il dispositivo pfSense non sembra averne). 

2. Feature Engineering Aggiornata (`build_daily_features`): 

    - Prima: Restituiva un dizionario (1 riga) per l'intera cartella. 
    - Ora: Raggruppa i dati interni al CSV per `device_id`. Restituisce una lista di dizionari (N righe), una per ogni dispositivo trovato in quel giorno. 

3. Modificato il Training Loop: 

    - Il raggruppamento dei dati per il training ora avviene sulla chiave composta `(source, device_id)`. 
    - Ogni dispositivo fisico ha la propria Isolation Forest e le sue statistiche storiche dedicate. 

4. Altri Fix: 

    - I file CSV di output ora includono la colonna `device_id` per distinguere le anomalie. 

5. Esecuzione automatizzata: 
    - Automatizzata l'analisi attraverso uno script Bash similmente a come fatto per la versione precedente. 

## Risultato

Ora l'analisi è specifica e i report indicano esattamente quale dispositivo ha rilevato anomalie. 