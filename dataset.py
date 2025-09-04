# In a file like dataset.py

import torch
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Assuming your dataset returns a dictionary with the SMILES string
        # Modify 'text_column_name' to the actual column name of SMILES strings in your dataset
        smiles_text = self.ds[idx]['smiles'] 
        
        tokens = self.tokenizer.encode(smiles_text).ids
        num_padding_tokens = self.seq_len - len(tokens) - 2 # For SOS and EOS

        if num_padding_tokens < 0:
            raise ValueError("SMILES string is longer than sequence length")

        # Input to the decoder
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # The label is the same as the input but shifted to the left
        label = torch.cat([
            torch.tensor(tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * (num_padding_tokens + 1), dtype=torch.int64) # +1 because we removed SOS
        ], dim=0)
        
        # The mask for the decoder's self-attention
        # This is a causal mask to prevent it from seeing future tokens
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            "decoder_input": decoder_input, # (seq_len)
            "decoder_mask": decoder_mask,  # (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": smiles_text # Original text for reference
        }

def causal_mask(size):
    # Creates a square matrix where the upper triangle (above the diagonal) is zero.
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0