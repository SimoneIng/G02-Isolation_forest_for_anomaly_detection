# Obiettivo

Ottimizzare l'esecuzione processando una singola tipologia di dispositivo sorgente alla volta. 


## Modifiche apportate

1. Nuovo argomento CLI (`--source`):
    - Lo script ora accetta una stringa che funge da filtro.

2. Logica di filtraggio cartelle: 
    - Modificata la funzione `list_source_day_folders`che ora accetta il parametro `target_source`.
    - Durante la scansione delle directory, le cartelle che non iniziano con la stringa `target_source` vengono ignorate. 

3. Esecuzione automatizzata: 
    - Creato uno script Bash che itera sulla lista dei dispositivi ed esegue lo script per ognuno salvando i risultati in cartelle dedicate. 


## Risultato

L'algoritmo di ML rimane invariato rispetto alla versione base, ma l'architettura di esecuzione ora è modulare. I modelli di IF sono addestrati specificatamente sui dati di quella sorgente.