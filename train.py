# train.py

from model import build_transformer # Your modified decoder-only model.py
from dataset import SmilesDataset, causal_mask # The new dataset class from dataset.py
from config import get_config # Your config file for hyperparameters

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import warnings
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import CharLevel
from tokenizers.trainers import CharLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, tokenizer, max_len, device):
    """
    Generates a sequence token by token using a greedy approach.
    """
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Start with the Start-Of-Sequence token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type(torch.int64).to(device)
    
    while True:
        # Stop if the sequence has reached max length
        if decoder_input.size(1) == max_len:
            break

        # Build a causal mask for the generated sequence
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        
        # Calculate the model's output
        out = model(decoder_input, decoder_mask) # model.forward pass

        # Get the next token by selecting the most probable one
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        
        # Append the new token to the sequence
        decoder_input = torch.cat([
            decoder_input, 
            torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)
        ], dim=1)

        # Stop if the End-Of-Sequence token is generated
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Runs validation by generating a few sample molecules.
    """
    model.eval() # Set the model to evaluation mode
    count = 0
    
    with torch.no_grad():
        for i in range(num_examples):
            print_msg("-" * 80)
            # Generate a molecule from scratch
            model_out = greedy_decode(model, tokenizer, max_len, device)
            model_out_text = tokenizer.decode(model_out.cpu().numpy())
            print_msg(f"Generated Sample {i+1}: {model_out_text}")

    model.train() # Set model back to training mode

def get_or_build_tokenizer(config, ds):
    """
    Builds a character-level tokenizer for the SMILES dataset.
    """
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        print("Building a new tokenizer...")
        tokenizer = Tokenizer(CharLevel()) # CharLevel is ideal for SMILES
        tokenizer.pre_tokenizer = Whitespace()
        trainer = CharLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        # Create an iterator for your text data
        def get_all_sentences():
            # IMPORTANT: Change 'text_column_name' to the actual column name of your SMILES strings
            for item in ds:
                yield item['smiles'] 

        tokenizer.train_from_iterator(get_all_sentences(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print("Loading existing tokenizer.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Loads the dataset, builds the tokenizer, and creates dataloaders.
    """
    # Load your SMILES dataset
    ds_raw = load_dataset(config['datasource'], split='train') 

    # Build a single tokenizer
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    # Split data into training and validation sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = SmilesDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = SmilesDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True) # Batch size 1 for validation

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_size):
    """
    Builds the decoder-only transformer model.
    """
    model = build_transformer(vocab_size, config["seq_len"], d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model

def train_model(config):
    """
    The main training loop.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    # Add your model preloading logic here if needed...

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Single forward pass through the decoder-only model
            proj_output = model(decoder_input, decoder_mask)
            
            # Calculate loss
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss to Tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save model checkpoint
        # Add your model saving logic here...

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)