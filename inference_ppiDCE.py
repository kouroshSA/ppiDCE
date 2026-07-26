#!/usr/bin/env python3
"""
inference_ppiDCE.py

Inference script for ppiDCE cross-encoder PPI classifier.

Inputs
------
CSV with at least 2 columns: seq1, seq2 (any other column, e.g. a label, is
ignored).

Outputs
-------
CSV with columns: seq1, seq2, Prediction, Probability_Friends, Probability_Enemies
(matches the ppiYYD / ppiBTEP inference convention).

Usage example:
    python inference_ppiDCE.py \
      --model_path path/to/ppiDCE_final.pth \
      --model_config facebook/esm1b_t33_650M_UR50S \
      --input_file test.csv \
      --output_file preds.csv \
      --batch_size 4 \
      --max_length 1024 \
      --device cuda
"""
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import EsmConfig, EsmTokenizer, EsmModel
from tqdm import tqdm

class PPICrossDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq1, seq2 = self.df.iloc[idx, 0], self.df.iloc[idx, 1]
        enc = self.tokenizer(
            seq1, seq2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0)
        }

class ppiDCE(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.esm = EsmModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls_vec)
        return self.classifier(x)


def get_device(device_str):
    if device_str.lower() == 'cpu':
        return torch.device('cpu'), None
    if ',' in device_str:
        devs = [d.strip() for d in device_str.split(',')]
        dev0 = devs[0]
        ids = [int(d.split(':')[-1]) for d in devs]
        return torch.device(dev0), ids
    return torch.device(device_str), None


def main():
    parser = argparse.ArgumentParser(description='Inference with ppiDCE model')
    parser.add_argument('--model_path', required=True, help='Path to ppiDCE checkpoint (.pth)')
    parser.add_argument('--model_config', required=True, help='ESM model name or local path')
    parser.add_argument('--input_file', required=True, help='CSV file with seq1, seq2')
    parser.add_argument('--output_file', required=True, help='CSV to save predictions')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Transformer layers the checkpoint was trained with. REQUIRED for '
                             'a from-scratch model with fewer layers than the pretrained config, '
                             'otherwise the extra layers load as random weights (silent, wrong).')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # device
    device, device_ids = get_device(args.device)

    # tokenizer + config
    tokenizer = EsmTokenizer.from_pretrained(args.model_config)
    config = EsmConfig.from_pretrained(args.model_config)
    if args.num_layers is not None:
        config.num_hidden_layers = args.num_layers
        print(f"Overriding config to {args.num_layers} transformer layers.")

    # dataset + loader
    ds = PPICrossDataset(args.input_file, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # model init
    model = ppiDCE(config, num_labels=2)

    # load checkpoint with filtering to avoid mismatched keys
    ckpt = torch.load(args.model_path, map_location='cpu')
    model_state = model.state_dict()
    filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_state and v.size() == model_state[k].size()}
    dropped = [k for k in ckpt if k not in filtered_ckpt]
    if dropped:
        print(f"WARNING: {len(dropped)}/{len(ckpt)} checkpoint keys NOT loaded (name/shape mismatch) "
              f"- check --num_layers matches training. First: {dropped[:3]}")
    model_state.update(filtered_ckpt)
    model.load_state_dict(model_state)

    if device_ids:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.eval()

    preds, probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Infer'):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            logits = model(input_ids, attn)
            p = nn.functional.softmax(logits, dim=1)
            pred = p.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            probs.extend(p.cpu().tolist())

    # assemble output (ppiYYD / ppiBTEP convention: seq1, seq2, Prediction,
    # Probability_Friends, Probability_Enemies)
    label_map = {0: 'enemies', 1: 'friends'}
    src = pd.read_csv(args.input_file)
    out = pd.DataFrame({
        'seq1': src.iloc[:, 0],
        'seq2': src.iloc[:, 1],
        'Prediction': [label_map[p] for p in preds],
        'Probability_Friends': [p[1] for p in probs],
        'Probability_Enemies': [p[0] for p in probs],
    })
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False)
    print(f"Saved inference results to {args.output_file}")

if __name__ == '__main__':
    main()
