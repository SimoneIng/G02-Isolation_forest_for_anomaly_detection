# Progetto IOT
Analisi di anomalie di rete con Isolation Forest su dispositivi di rete (firewall fisici/virtuali). 

## Materiale fornito in Input
Lo script `anomaly-test-3.1_mod.py` implementa un sistema di rilevamento anomalie basato sull'algoritmo **Isolation Forest**. L'obiettivo è analizzare il traffico di rete proveniente da diverse fonti eterogenee (firewall fisici, virtuali ecc) per identificare comportamenti anomali.

###  Dataset

Il dataset è raggruppato in cartelle che indicano la tipologia di dispositivo da cui sono stati raccolti i dati e la giornata in cui sono stati raccolti.
Ogni file csv è composto da righe che indicano un evento di rete. 


### Flusso logico di esecuzione

1. **Lettura dei dati:** Legge file di log organizzati in cartelle giornaliere.

2. **Feature Engineering:** Aggrega i dati grezzi trasformandoli in metriche giornaliere per ogni "source" (es. numero eventi, totale byte, IP unici, rateo di azioni "deny/accept").

3. **Statistica Robusta:** Confronta i dati del giorno corrente con uno storico (training set) utilizzando metriche robuste (Mediana e IQR - Interquartile Range) invece di Media e Deviazione Standard, per essere meno sensibile agli outlier storici.

4. **Machine Learning:** Utilizza l'algoritmo di ML *Isolation Forest* per rilevare anomalie.

5. **Spiegabilità:** Non dice solo "c'è un'anomalia", ma spiega *perché* (es. "il traffico byte è salito di 3 deviazioni standard rispetto al solito").


### Struttura del file

Il codice segue questa struttura logica:

1. **Imports & Setup GPU:** Importa le librerie necessarie cercando di impostare anche accelerazione GPU. 

2. **Configurazione (`ALIASES`, `ACTION_VALUES`):** Definisce dizionari per normalizzare i nomi delle colonne (es. mappando "srcip", "source_ip" tutti su "src_ip") e i valori delle azioni.

3. **Utility (Date e I/O):** Funzioni per il parsing delle date dai nomi delle cartelle e per la lettura selettiva dei file (legge solo le colonne utili).

4. **Canonicalizzazione:** Logica per unificare i nomi delle colonne eterogenei dei vari log.

5. **Feature Engineering (`build_daily_features`):** Trasformazione da "milioni di righe di log" a "una riga di statistiche per giorno".

6. **Logica Statistica (`robust_...`, `score_...`):** Implementazione del calcolo Z-score e del sistema di "voto" per definire la gravità dell'anomalia.

7. **Isolation Forest:** Wrapper per l'algoritmo di ML.

8. **Runner (`run`):** La funzione che orchestra tutto il flusso (ciclo sui giorni, split train/test, salvataggio risultati).

9. **Main:** Gestione degli argomenti da riga di comando.

### Elenco e descrizione delle funzioni

- **`list_source_day_folders(...)`**: Scansiona la directory di input cercando cartelle nel formato `nome-GG.MM.AAAA` e le filtra in base alle date di start/end richieste.

- **`read_selective(...)`**: Legge un file (Parquet o CSV) caricando **solo** le colonne definite nella configurazione, risparmiando memoria.

- **`canonicalize_columns(df)`**: Rinomina le colonne del DataFrame usando il dizionario `ALIASES` per avere nomi standard (es. `timestamp`, `src_ip`, `bytes`).

- **`build_daily_features(df, source, day)`**: Prende il DataFrame dei log grezzi di un giorno e calcola:
    - Totale eventi e byte.
    - Cardinalità (quanti IP sorgente unici, quanti host unici, ecc.).
    - Conteggi e percentuali delle `action` (es. % di traffico "deny").

- **`training_stats(train_df)`**: Calcola la Mediana e la Sigma Robusta (derivata dall'IQR) per ogni feature dello storico di training.

- **`robust_deviation(x, med, iqr_scaled)`**: Calcola lo Z-score: quanto il valore odierno si discosta dalla mediana storica.

- **`score_and_explain(...)`**:
    - Confronta i valori di oggi con le statistiche storiche.
    - Assegna "voti": 2 punti se Z > 3.5 (deviazione forte), 1 punto se Z > 2.5 (deviazione media).
    - Restituisce uno score aggregato e una lista di spiegazioni testuali.

- **`iforest_score_from_training(...)`**: Addestra un modello Isolation Forest sui dati storici e restituisce l'anomaly score per il giorno corrente.

- **`run(...)`**: La funzione principale che esegue il loop su tutti i giorni, gestisce la finestra mobile (o l'holdout) per il training, calcola le anomalie e scrive i CSV di output.

### Output

Lo script crea una cartella di output (definita da `--out` + timestamp) contenente 4 file CSV principali:

1. **`daily_features.csv`**:
    - Ogni riga è un giorno.
    - Colonne: `source`, `date`, `total_events`, `bytes_total`, `unique_src_ip`, `action_deny_rate`, ecc.

2. **`daily_anomalies.csv`**:
    - Contiene i risultati del rilevamento.

    - **Label**: `NORMAL`, `LOW_VOLUME`, `MEDIUM`, `STRONG`.
    - **Score**: Punteggio di anomalia basato sui voti statistici.
    - **Explain_top**: Una stringa che indica cosa non va (es. `bytes_total:+4.2(UP), unique_dst_ip:-2.8(DOWN)`).

3. **`daily_anomaly_explanations.csv`**:
    - File molto lungo.
    - Per ogni giorno anomalo, elenca *tutte* le feature analizzate, il loro valore, la mediana storica e lo Z-score.
    - Utile per analisi approfondite.

4. **`daily_iforest_scores.csv`** (Se viene utilizzato `-iforest`):
    - Contiene lo score specifico dell'Isolation Forest. Più lo score è negativo, più è anomalo.


###  Considerazioni 

- Il dataset viene caricato ed elaborato tutto in una volta sola.
- Lo script non tiene conto della diversità dei dispositivi forniti in input.


## Obiettivo
Raffinare l'analisi in due step: 
1. Considerare nell'addestramento soltanto una tipologia di dispositivo alla volta, senza mischiare i dati (es. fortigate). 

2. Individuare un campo che indichi univocamente ogni dispositivo di rete che ha raccolto i dati ed eseguire l'Isolation Forest considerando soltanto i dati raccolti da esso. 

3. Confrontare brevemente i risultati.

## Output

All'interno della cartella di output sono presenti due sottocartelle. 

1. In `output/prima-valutazione` è stato spostato lo script python aggiornato e le sottocartelle che racchiudono i risultati dell'esecuzione dello script per ogni tipologia di dispositivo. 

    Inoltre, è presente uno script bash che esegue l'analisi su ogni dispositivo chiamando automaticamente lo script python. 

    I cambiamenti sono documentati in `changelog_V1`. 

2. In `output/seconda-valutazione` è stato spostato lo script aggiornato per riconoscere i dispositivi univocamente. 

    Anche in questa sottocartella è presente uno script bash che esegue l'analisi in automatico.

    Tutti i cambiamenti rispetto alla versione precedente sono documentati in `changelog_V2`.

Se si vuole provare ad eseguire questa valutazione, spostare gli script nella directory principale del progetto.