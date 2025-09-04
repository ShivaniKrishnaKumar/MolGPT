# config.py

from pathlib import Path

def get_config():
    return {
        # --- Training Hyperparameters ---
        "batch_size": 16, # Adjust based on your GPU memory
        "num_epochs": 50, # More epochs might be needed for generation tasks
        "lr": 1e-4,
        "seq_len": 256,   # CHANGED: Adjust to the max length of SMILES strings in your dataset
        "d_model": 512,
        "N": 6,           # Number of decoder blocks
        "h": 8,           # Number of attention heads
        "dropout": 0.1,
        "d_ff": 2048,

        # --- Data & Model Paths ---
        "datasource": "basu369victor/zinc250k", # CHANGED: Use your Hugging Face dataset name or local path
        "model_folder": "smiles_weights",         # CHANGED: More descriptive folder name
        "model_basename": "mol_gen_tmodel_",      # CHANGED: More descriptive model name
        "preload": "latest",
        "tokenizer_file": "smiles_tokenizer.json",# CHANGED: Simplified for a single tokenizer
        "experiment_name": "runs/mol_gen_model"   # CHANGED: More descriptive experiment name
    }

# This function works as-is but will use the updated keys from the config
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# This function also works as-is
def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    weights_files = list(Path(model_folder).glob(f"{model_basename}*"))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])