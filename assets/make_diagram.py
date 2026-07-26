#!/usr/bin/env python3
"""
make_diagram.py — render the ppiDCE ASCII architecture/workflow diagram to a PNG.

Light monospace box-drawing on a dark background, sections A-D — matching the
ppiYYD / ppiBTEP diagram style. The diagram is built programmatically so the
boxes and the C/D two-column pipeline stay aligned.

    python assets/make_diagram.py           # -> assets/ppiDCE.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE = 26
BG = (18, 20, 28)
FG = (222, 226, 234)
DIM = (150, 158, 176)
ACCENT = (126, 186, 240)
PAD = 34
LINE_SPACING = 1.34
PUB = True

W = 80                      # nominal diagram width in cells
CD_LEFT = 41                # column of the C/D divider


def rule(title):
    body = f"── {title} "
    return body + "─" * max(0, W - len(body))


def dbox(interior, width, indent=2):
    """Double-border box (╔╗╚╝ ║) around interior lines, each padded to width."""
    pad = " " * indent
    top = pad + "╔" + "═" * (width + 2) + "╗"
    bot = pad + "╚" + "═" * (width + 2) + "╝"
    mid = [pad + "║ " + s.ljust(width) + " ║" for s in interior]
    return [top] + mid + [bot]


def sbox(interior, width, indent=2):
    """Single-border box (┌┐└┘ │)."""
    pad = " " * indent
    top = pad + "┌" + "─" * (width + 2) + "┐"
    bot = pad + "└" + "─" * (width + 2) + "┘"
    mid = [pad + "│ " + s.ljust(width) + " │" for s in interior]
    return [top] + mid + [bot]


def cd_row(left, right):
    return "  " + left.ljust(CD_LEFT - 2) + "│  " + right


def build():
    L = []
    add = L.extend

    # ---- header ----
    add(dbox([
        "    ppiDCE : Dual Cross-Encoder for PPI Classification",
        "    Foundation : ESM-1b  (facebook/esm1b_t33_650M_UR50S)",
    ], 72, indent=0))
    L.append("")

    # ---- A. input ----
    add([rule("A. CROSS-ENCODING INPUT STRATEGY"), ""])
    add([
        "  Protein A (Seq_A)                        Protein B (Seq_B)",
        "  [ M  A  E  G  V  L  K ... ]              [ M  K  T  V  L  I  P ... ]",
        "        │                                        │",
        "        └────────────────┬───────────────────────┘",
        "                         │  sentence-pair concatenation",
        "                         ▼",
    ])
    add(sbox([
        "[CLS]  Seq_A tokens ...  [SEP]  Seq_B tokens ...  [EOS]",
        "                max_length = 1024 tokens",
    ], 58, indent=2))
    add(["          ↑ full residue-to-residue cross-attention ↑", ""])

    # ---- B. architecture ----
    add([rule("B. MODEL ARCHITECTURE"), ""])
    add(sbox(["Token Embeddings + Positional Embeddings"], 44, indent=17))
    add(["                                  │",
         "                                  ▼"])
    backbone = ["ESM-1b Transformer Backbone", ""]
    backbone += ["  ┌" + "─" * 66 + "┐"]
    backbone += ["  │ Layer 1  — Multi-Head Self-Attention  +  FFN" + " " * 20 + "│"]
    backbone += ["  │   Seq_A residues  ↔  Seq_B residues  (fully bidirectional)  │"]
    backbone += ["  └" + "─" * 66 + "┘"]
    backbone += ["                             . . ."]
    backbone += ["  ┌" + "─" * 66 + "┐"]
    backbone += ["  │ Layer N  — Self-Attn + FFN  (N per --num_layers, e.g. 6)" + " " * 8 + "│"]
    backbone += ["  │   Seq_A residues  ↔  Seq_B residues  (fully bidirectional)  │"]
    backbone += ["  └" + "─" * 66 + "┘"]
    add(dbox(backbone, 72, indent=2))
    add(["                                  │  last_hidden_state[ : , 0 , : ]",
         "                                  ▼"])
    add(sbox(["[CLS] Token Vector", "     dim = 1280"], 26, indent=17))
    add(["                                  │",
         "                                  ▼"])
    add(dbox([
        "                     Classification Head",
        "     Dropout(p=0.1)  ──▶  Linear(1280 → 2)  ──▶  Softmax",
    ], 72, indent=2))
    add(["                     │                          │",
         "                     ▼                          ▼"])
    add([
        "        ┌──────────────────────┐    ┌──────────────────────┐",
        "        │  prob_class_0        │    │  prob_class_1        │",
        "        │  Non-Interacting     │    │  Interacting         │",
        "        └──────────────────────┘    └──────────────────────┘",
        "",
    ])

    # ---- C / D pipelines ----
    header = "── C. TRAINING PIPELINE " + "─" * (CD_LEFT - len("── C. TRAINING PIPELINE ") - 2)
    header = header.ljust(CD_LEFT) + "┬─ D. INFERENCE PIPELINE " + "─" * 14
    L.append(header)
    L.append(cd_row("", ""))
    rows = [
        ("Input: seq1, seq2, label (CSV)", "Input: seq1, seq2 (CSV)"),
        ("      │", "      │"),
        ("PPICrossDataset + DataLoader", "PPICrossDataset + DataLoader"),
        ("      │", "      │"),
        ("ppiDCE  forward pass", "Load .pth checkpoint (--num_layers N)"),
        ("      │", "      │"),
        ("CrossEntropyLoss", "ppiDCE  forward pass"),
        ("      │", "      │"),
        ("AdamW + LR sched (const|warmup-cos)", "Softmax  probabilities"),
        ("      │", "      │"),
        ("→ ppiDCE_epoch{N}.pth", "Output: pred_label, prob_0, prob_1"),
        ("      │", ""),
        ("per-epoch PRS/RRS AUC + best-F1", "(LES-wrapper.py scores every epoch)"),
    ]
    for lft, rgt in rows:
        L.append(cd_row(lft, rgt))
    L.append("─" * CD_LEFT + "┴" + "─" * (W - CD_LEFT - 1))
    return L


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "ppiDCE.png")
    lines = build()
    font = ImageFont.truetype(FONT, FONT_SIZE)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cell_w = probe.textlength("M", font=font)
    line_h = int(FONT_SIZE * LINE_SPACING)
    width = int(max(len(l) for l in lines) * cell_w) + 2 * PAD
    height = line_h * len(lines) + 2 * PAD

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = PAD + i * line_h
        s = line.strip()
        if s.startswith("──") or s.startswith("───"):
            colour = ACCENT
        elif s.startswith("╔") or s.startswith("║") or s.startswith("╚"):
            colour = FG
        elif s.startswith("(") or s.startswith("NOTE"):
            colour = DIM
        else:
            colour = FG
        draw.text((PAD, y), line, font=font, fill=colour)

    img.save(out_path)
    print(f"wrote {out_path}  ({width}x{height}, {len(lines)} lines)")


if __name__ == "__main__":
    main()
