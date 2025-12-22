from pathlib import Path

def save_execution_params(args, output_dir: Path):
    """Salva i parametri di esecuzione in un file nella directory di output."""
    from datetime import datetime
    import os
    import json
    
    
    # Crea il nome del file con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    params_file = output_dir / f"params_{script_name}_{timestamp}.json"
    
    # Converti args in dizionario
    params = vars(args)
    
    # Aggiungi informazioni extra
    execution_info = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "parameters": params
    }
    
    # Assicurati che la directory esista
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scrivi in formato JSON per facile lettura/parsing
    with open(params_file, 'w') as f:
        json.dump(execution_info, f, indent=2)
    
    print(f"Parametri salvati in: {params_file}")
    return params_file