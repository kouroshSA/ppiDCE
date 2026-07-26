#!/usr/bin/env python3
"""
ppiDCE: Dual Cross-Encoder for PPI Classification.

Dependencies
------------
    conda create -n esm python=3.10 && conda activate esm
    pip install torch        # pick the CUDA build that matches your driver
    pip install transformers pandas tqdm

(Both training and inference use only the transformers and pandas packages
beyond PyTorch.)
"""
import argparse
import math
import os
import subprocess
import sys
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
                        help='Learning rate. With --lr_schedule warmup_cosine this is the '
                             'PEAK reached at the end of warmup.')
    # LR schedule (ported from ppiYYD; stepped PER ITERATION, mirrors nanoGPT get_lr)
    parser.add_argument('--lr_schedule',      type=str, default='constant',
                        choices=['constant', 'warmup_cosine'],
                        help="LR schedule, stepped per optimizer step. 'constant' (default) "
                             "holds --learning_rate. 'warmup_cosine' ramps 0->peak over the "
                             "warmup, then cosine-decays peak->--min_lr over the rest.")
    parser.add_argument('--warmup_ratio',     type=float, default=0.1,
                        help='warmup_cosine: warmup length as a fraction of total steps '
                             '(default 0.1 = 1 epoch of a 10-epoch run). Overridden by '
                             '--warmup_steps.')
    parser.add_argument('--warmup_steps',     type=int, default=None,
                        help='warmup_cosine: explicit warmup length in iterations '
                             '(overrides --warmup_ratio).')
    parser.add_argument('--min_lr',           type=float, default=0.0,
                        help='warmup_cosine: floor the cosine decays to (e.g. 2e-6).')
    parser.add_argument('--max_length',       type=int,   default=1024,
                        help='Max total tokens (seq1+seq2+special)')
    # Runtime
    parser.add_argument('--output_dir',       type=str, default='./',
                        help='Directory to save checkpoints and final model')
    parser.add_argument('--device',           type=str, default='cuda', choices=['cpu','cuda'],
                        help='Device for training')
    # Per-epoch holdout eval on PRS/RRS reference sets (in-process, no GPU contention)
    parser.add_argument('--eval_prs',         type=str, default=None,
                        help='PRS holdout CSV (seq1,seq2) scored after each epoch for AUC/Best-F1.')
    parser.add_argument('--eval_rrs',         type=str, default=None,
                        help='RRS holdout CSV (seq1,seq2) scored after each epoch for AUC/Best-F1.')
    parser.add_argument('--eval_dir',         type=str, default=None,
                        help='Directory for metrics_by_epoch.csv (default: output_dir).')
    parser.add_argument('--roc_script',       type=str, default=None,
                        help='Path to roc_analysis_color_threshold_F1e.py. When set (and the '
                             'PRS/RRS eval hook is active) a ROC + Best-F1 figure is rendered '
                             'after every epoch from that epoch\'s probabilities. Defaults to '
                             'the copy sitting next to this script; pass "none" to disable.')
    return parser.parse_args()


def _write_prob_csv(path, prs, rrs):
    """Write the 2-column PRS/RRS probability CSV that roc_analysis_*.py consumes.

    Columns are ragged-tolerant: the reader takes column 0 as a PRS probability
    and column 1 as an RRS probability, skipping empties, so unequal set sizes
    are fine.
    """
    with open(path, 'w') as fh:
        fh.write('PRS,RRS\n')
        for i in range(max(len(prs), len(rrs))):
            p = f'{prs[i]:.6f}' if i < len(prs) else ''
            r = f'{rrs[i]:.6f}' if i < len(rrs) else ''
            fh.write(f'{p},{r}\n')


def _read_pairs(csv_path):
    """Read a headerless-or-headered 2-column (seq1,seq2) reference CSV -> (list1, list2)."""
    df = pd.read_csv(csv_path, header=None)
    if str(df.iloc[0, 0]).strip().lower() in ('seq1', 'sequence1', 'seq_a', 'protein1'):
        df = df.iloc[1:].reset_index(drop=True)
    return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1].astype(str).tolist()


@torch.no_grad()
def _score_pairs(model, tokenizer, s1, s2, max_length, device, batch_size=8):
    """P(interaction) = softmax(logits)[1] for each (seq1,seq2) pair, cross-encoded."""
    model.eval()
    out = []
    for i in range(0, len(s1), batch_size):
        enc = tokenizer(s1[i:i + batch_size], s2[i:i + batch_size], truncation=True,
                        padding='max_length', max_length=max_length, return_tensors='pt')
        logits = model(enc.input_ids.to(device), enc.attention_mask.to(device))
        out.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
    return out


def _auc_bestf1(prs, rrs):
    """ROC-AUC and Best-F1 (scanning every unique score) for PRS=1 / RRS=0."""
    import numpy as np
    from sklearn.metrics import roc_curve, auc as sk_auc, f1_score
    probs = np.array(prs + rrs, dtype=float)
    labels = np.array([1] * len(prs) + [0] * len(rrs))
    fpr, tpr, _ = roc_curve(labels, probs)
    a = float(sk_auc(fpr, tpr))
    best_f1, best_t = -1.0, 0.0
    for t in np.unique(probs):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f >= best_f1:
            best_f1, best_t = float(f), float(t)
    return a, best_f1, best_t

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

    # per-iteration LR schedule (mirrors nanoGPT get_lr(it) and ppiYYD): linear
    # warmup 0->peak, then cosine decay peak->min_lr. Stepped once per optimizer step.
    scheduler = None
    if args.lr_schedule == 'warmup_cosine':
        total_steps = args.epochs * len(train_loader)
        warmup = (args.warmup_steps if args.warmup_steps is not None
                  else int(round(args.warmup_ratio * total_steps)))
        peak, floor = args.learning_rate, args.min_lr

        def lr_lambda(step):                       # returns lr(step)/peak
            if step < warmup:
                return (step + 1) / (warmup + 1)
            dr = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            coeff = 0.5 * (1.0 + math.cos(math.pi * dr))    # 1 -> 0
            return (floor + coeff * (peak - floor)) / peak

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"LR schedule: warmup_cosine | peak={peak:g} min={floor:g} "
              f"warmup={warmup} steps ({warmup/len(train_loader):.2f} epoch) "
              f"total={total_steps} steps (per-iteration)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the per-epoch ROC/F1 plotting script (default: alongside this file).
    roc_script = args.roc_script
    if roc_script is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'roc_analysis_color_threshold_F1e.py')
        roc_script = cand if os.path.exists(cand) else None
    elif roc_script.lower() == 'none':
        roc_script = None
    if roc_script and not os.path.exists(roc_script):
        print(f"WARNING: --roc_script {roc_script} not found; per-epoch ROC figures disabled")
        roc_script = None

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
            if scheduler is not None:
                scheduler.step()               # per-iteration LR update
            total_loss += loss.item()
        print(f"Train loss: {total_loss/len(train_loader):.4f}  (lr={optimizer.param_groups[0]['lr']:.3e})")

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

        # Per-epoch holdout eval on PRS/RRS (what we actually care about): AUC + Best-F1.
        if args.eval_prs and args.eval_rrs:
            s1p, s2p = _read_pairs(args.eval_prs)
            s1r, s2r = _read_pairs(args.eval_rrs)
            prs = _score_pairs(model, tokenizer, s1p, s2p, args.max_length, device)
            rrs = _score_pairs(model, tokenizer, s1r, s2r, args.max_length, device)
            auc_v, f1_v, thr_v = _auc_bestf1(prs, rrs)
            print(f"[epoch {epoch}] holdout PRS/RRS: AUC={auc_v:.4f} best_f1={f1_v:.4f} "
                  f"thr={thr_v:.4f}  (PRS {len(prs)}, RRS {len(rrs)})")
            ed = args.eval_dir or args.output_dir
            os.makedirs(ed, exist_ok=True)
            mp = os.path.join(ed, 'metrics_by_epoch.csv')
            new = not os.path.exists(mp)
            with open(mp, 'a') as fh:
                if new:
                    fh.write('epoch,auc,best_f1,threshold\n')
                fh.write(f'{epoch},{auc_v:.6f},{f1_v:.6f},{thr_v:.6f}\n')

            # Per-epoch ROC + Best-F1 figure, rendered from the probabilities we
            # just computed (no extra forward pass).
            prob_csv = os.path.join(ed, f'epoch{epoch}_PRS-RRS_probabilities.csv')
            _write_prob_csv(prob_csv, prs, rrs)
            if roc_script:
                try:
                    subprocess.run(
                        [sys.executable, roc_script, '--input_csv', prob_csv,
                         '--output_file', os.path.join(ed, f'roc_epoch{epoch}.png')],
                        check=True, capture_output=True, text=True)
                    print(f"[epoch {epoch}] ROC figure: {os.path.join(ed, f'roc_epoch{epoch}.png')}")
                except subprocess.CalledProcessError as e:
                    print(f"[epoch {epoch}] WARNING: ROC script failed: {e.stderr.strip()}")

            model.train()

    # Final save
    final_model = os.path.join(args.output_dir, "ppiDCE_final.pth")
    torch.save(model.module.state_dict() if hasattr(model,'module') else model.state_dict(), final_model)
    print(f"Saved final model: {final_model}")

if __name__ == '__main__':
    main()
