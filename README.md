# ppiDCE

A dual cross-encoder for binary protein-protein interaction (PPI) classification, built on ESM-1b ([Rives et al., 2021](https://doi.org/10.1073/pnas.2016239118)).

![ppiDCE Architecture](assets/ppiDCE.png)

## Overview

ppiDCE repurposes ESM-1b -- a single-sequence masked language model with no native PPI capability -- for protein-protein interaction prediction by exploiting its tokenizer's sentence-pair encoding mode. Both protein sequences are concatenated into a single input as `[CLS] Seq_A [SEP] Seq_B [EOS]`, enabling full bidirectional cross-attention between the two sequences at every transformer layer. The `[CLS]` token representation from the final layer captures joint inter-protein features and is passed through a dropout + linear classification head to produce binary interaction predictions with softmax probabilities.

The model was developed for the *Prochlorococcus marinus* MED4 interactome, where it serves as one component of a tri-model consensus framework (alongside [ppiGPLM](https://github.com/kouroshSA/ppiGPLM) and [ppiBTEP](https://github.com/kouroshSA/ppiBTEP)) for computational PPI screening.

## Architecture

| Parameter | Value |
|-----------|-------|
| Foundation | ESM-1b (facebook/esm1b_t33_650M_UR50S) |
| Strategy | Cross-encoding (sentence-pair) |
| Layers | 12 (configurable) |
| Classification | [CLS] -> Dropout(0.1) -> Linear -> 2 |
| Max sequence length | 1,024 tokens |
| Optimizer | AdamW (lr = 2 x 10^-5) |
| Loss | Cross-Entropy |

### Cross-Encoding vs Single-Sequence

Unlike the original ESM-1b which processes one protein at a time, ppiDCE feeds both proteins as a single concatenated input. This enables inter-protein residue-residue attention at every transformer layer -- the most expressive strategy for modeling pairwise interactions, at the cost of O((n+m)^2) attention complexity.

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/kouroshSA/ppiDCE.git
cd ppiDCE

# Create a conda environment
conda create -n esm python=3.10
conda activate esm
pip install -r requirements.txt
```

## Repository Structure

```
ppiDCE/
|-- train_ppiDCE.py                    # Training script
|-- inference_ppiDCE.py                # Batch inference script
|-- roc_analysis_color_threshold_F1e.py  # ROC curve analysis with F1 optimization
|-- assets/
|   |-- ppiDCE.png                     # Model architecture diagram
|   |-- ppiDCE_architecture.svg        # Detailed architecture flow (SVG)
|   +-- ppiDCE_architecture.png        # Detailed architecture flow (PNG)
|-- requirements.txt
|-- LICENSE
+-- README.md
```

## Usage

### Data Format

Training and inference use CSV files with columns: `protein1_seq, protein2_seq, label`

- `protein1_seq`, `protein2_seq`: Amino acid sequences
- `label`: `0` (non-interacting) or `1` (interacting)

For inference-only input, only the first two columns are required.

### Training

```bash
# Train from scratch with 12 layers
python train_ppiDCE.py \
    --train_file train.csv \
    --val_file val.csv \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --from_scratch \
    --num_layers 12 \
    --epochs 10 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_length 1024 \
    --output_dir ./out \
    --device cuda
```

#### Key training options

- `--from_scratch`: Initialize ESM backbone with random weights (recommended for PPI)
- `--num_layers N`: Set total transformer layers when training from scratch
- `--freeze_layers N`: Freeze bottom N layers during fine-tuning
- `--add_layers N`: Append extra transformer layers on top
- `--checkpoint path.pth`: Resume from a saved checkpoint
- `--suppress_warnings`: Suppress tokenizer truncation warnings

### Inference

```bash
python inference_ppiDCE.py \
    --model_path out/ppiDCE_epoch8.pth \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --input_file test_pairs.csv \
    --output_file predictions.csv \
    --batch_size 4 \
    --max_length 1024 \
    --device cuda
```

Output CSV columns: `seq1, seq2, pred_label, prob_0, prob_1`

### ROC Analysis

Evaluate model predictions using ROC curve analysis with threshold-colored visualization and F1 optimization:

```bash
python roc_analysis_color_threshold_F1e.py \
    --input_csv probabilities.csv \
    --output_file roc_curve.png
```

The input CSV should have two columns: PRS (positive) and RRS (random/negative) probability values.

## Architecture Diagram

See `assets/ppiDCE_architecture.svg` for a detailed flow diagram covering:
- **A.** Model architecture (cross-encoding strategy)
- **B.** Cross-encoding vs single-sequence comparison
- **C.** Training pipeline
- **D.** Inference pipeline

## Citation

If you use this software, please cite:

```
Daakour, S. et al. (2026).
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
