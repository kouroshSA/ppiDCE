#!/usr/bin/env python3
"""
ppiDCE: Dual Cross-Encoder for PPI Classification

***************DEPENDENCY INSTALLATION********************************************************************
Make a new conda env with python 3.10, e.g., $conda create -n esm python=3.10, then $conda activate esm
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas
pip install fair-esm #Install the ESM Library
pip install biopython
pip install transformers
conda install tqdm
********************************************************************************************************


This script trains or fine-tunes the ppiDCE model—a dual cross-encoder built on ESM—for protein-protein interaction classification.

Features:
 - Uses ESM tokenizer and model backbone
 - 3-column CSV input: protein1, protein2, label (0 or 1)
 - Optional training from scratch (`--from_scratch`) or loading pretrained ESM backbone
 - Specify total transformer layers with `--num_layers` when starting from scratch
 - Optional freezing of bottom transformer layers
 - Optional addition of extra transformer layers on top of base model
 - Optional suppressing of tokenizer warnings (`--suppress_warnings`)
 - Automatically extends positional embeddings if `--max_length` exceeds default model limit
 - Optional loading from checkpoint for continued training

Example:
```bash
# Run training from scratch with 12 layers + 2 appended layers, suppressing warnings:
python train_ppiDCE.py \
  --train_file train.csv \
  --val_file val.csv \
  --model_config facebook/esm1b_t33_650M_UR50S \
  --from_scratch \
  --num_layers 12 \
  --add_layers 2 \
  --suppress_warnings \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --max_length 2048 \
  --output_dir ./out \
  --device cuda
```
"""
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import EsmConfig, EsmTokenizer, EsmModel, logging as hf_logging
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train or fine-tune ppiDCE: dual cross-encoder PPI classifier.'
    )
    # Data
    parser.add_argument('--train_file',       type=str, required=True,
                        help='Path to training CSV: seq1,seq2,label')
    parser.add_argument('--val_file',         type=str, required=True,
                        help='Path to validation CSV: seq1,seq2,label')
    # Model
    parser.add_argument('--model_config',     type=str, required=True,
                        help='HuggingFace ESM model name or local path')
    parser.add_argument('--num_labels',       type=int, default=2,
                        help='Number of output labels (binary=2)')
    parser.add_argument('--from_scratch',     action='store_true',
                        help='Initialize ESM backbone randomly instead of loading pretrained')
    parser.add_argument('--num_layers',       type=int, default=None,
                        help='Total number of transformer layers when initializing from scratch')
    parser.add_argument('--freeze_layers',    type=int, default=0,
                        help='Number of bottom encoder layers to freeze (ignored if from_scratch)')
    parser.add_argument('--add_layers',       type=int, default=0,
                        help='Number of extra transformer layers to append')
    parser.add_argument('--suppress_warnings', action='store_true',
                        help='Suppress tokenizer truncation warnings')
    parser.add_argument('--checkpoint',       type=str, default=None,
                        help='Optional checkpoint (.pth) to load weights from')
    # Training
    parser.add_argument('--epochs',           type=int,   default=3,
                        help='Total training epochs')
    parser.add_argument('--batch_size',       type=int,   default=8,
                        help='Batch size for train/validation')
    parser.add_argument('--learning_rate',    type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length',       type=int,   default=1024,
                        help='Max total tokens (seq1+seq2+special)')
    # Runtime
    parser.add_argument('--output_dir',       type=str, default='./',
                        help='Directory to save checkpoints and final model')
    parser.add_argument('--device',           type=str, default='cuda', choices=['cpu','cuda'],
                        help='Device for training')
    return parser.parse_args()

class PPICrossDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq1, seq2, lbl = self.df.iloc[idx]
        enc = self.tokenizer(
            seq1, seq2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels':         torch.tensor(int(lbl), dtype=torch.long)
        }

class ppiDCE(nn.Module):
    def __init__(self, esm_model, num_labels=2):
        super().__init__()
        self.esm = esm_model
        hidden_size = esm_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        return self.classifier(x)


def main():
    args = parse_args()

    # Optionally suppress tokenizer warnings
    if args.suppress_warnings:
        hf_logging.set_verbosity_error()
        logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer & config
    tokenizer = EsmTokenizer.from_pretrained(args.model_config)
    config = EsmConfig.from_pretrained(args.model_config)

    # Set layers for scratch
    if args.from_scratch:
        if args.num_layers:
            config.num_hidden_layers = args.num_layers
        print(f"Initializing from scratch with {config.num_hidden_layers} layers")

    # Append layers
    if args.add_layers:
        config.num_hidden_layers += args.add_layers
        print(f"Total layers after appending: {config.num_hidden_layers}")

        # Load or init backbone with proper positional embeddings
    # First, adjust config for desired positional embeddings
    if args.from_scratch:
        # Build fresh model with config (including any num_layers modifications)
        esm_model = EsmModel(config)
        print("Initialized new ESM model from scratch.")
    else:
        # Instantiate model architecture with extended positional embeddings
        esm_model = EsmModel(config)
        # Load pretrained weights where shapes match
        print(f"Loading pretrained weights from {args.model_config} into extended model architecture...")
        pretrained = EsmModel.from_pretrained(args.model_config)
        pretrained_state = pretrained.state_dict()
        model_state = esm_model.state_dict()
        # Copy matching parameters
        for key, weight in pretrained_state.items():
            if key in model_state and pretrained_state[key].shape == model_state[key].shape:
                model_state[key] = weight
        esm_model.load_state_dict(model_state)
        print("Pretrained weights loaded for matching parameters.")

    # If args.max_length exceeds original model limit, ensure positional embeddings exist
    max_pos = esm_model.config.max_position_embeddings
    if args.max_length > max_pos:
        print(f"Extending positional embeddings from {max_pos} to {args.max_length}")
        old_embed = esm_model.embeddings.position_embeddings.weight.data
        new_embed = nn.Embedding(args.max_length, old_embed.size(1))
        # Copy old embeddings and init new ones
        new_embed.weight.data[:max_pos] = old_embed
        new_embed.weight.data[max_pos:] = old_embed.new_empty(args.max_length - max_pos, old_embed.size(1)).normal_(0.0, 0.02)
        esm_model.embeddings.position_embeddings = new_embed
        esm_model.config.max_position_embeddings = args.max_length

    # Dataset & loaders
    train_ds = PPICrossDataset(args.train_file, tokenizer, args.max_length)
    val_ds   = PPICrossDataset(args.val_file,   tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model instantiation
    model = ppiDCE(esm_model, num_labels=args.num_labels)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Freeze layers
    if not args.from_scratch and args.freeze_layers > 0:
        for p in model.esm.embeddings.parameters(): p.requires_grad=False
        for i in range(min(args.freeze_layers, len(model.esm.encoder.layer))):
            for p in model.esm.encoder.layer[i].parameters(): p.requires_grad=False
        print(f"Frozen bottom {args.freeze_layers} layers")

    model.to(device)
    if torch.cuda.device_count()>1 and device.type=='cuda': model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training & validation
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                loss = criterion(logits, batch['labels'].to(device))
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch['labels'].to(device)).sum().item()
                total += len(preds)
        print(f"Val loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"ppiDCE_epoch{epoch}.pth")
        torch.save(model.module.state_dict() if hasattr(model,'module') else model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_model = os.path.join(args.output_dir, "ppiDCE_final.pth")
    torch.save(model.module.state_dict() if hasattr(model,'module') else model.state_dict(), final_model)
    print(f"Saved final model: {final_model}")

if __name__ == '__main__':
    main()
