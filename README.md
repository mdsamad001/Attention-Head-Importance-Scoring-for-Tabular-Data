# Attention Head Importance Scoring and Dynamic Tuning for Tabular Transformers

This repository contains the code for reproducing the experiments in:

> **[Paper Title]**
> [Authors], [Venue, Year]

We investigate the importance of individual attention heads in transformer models for tabular data. Using the [TransTab](https://github.com/RyanWangZf/transtab) architecture, we adapt the head-importance metric $I_h$ from [Michel et al. (2019)](https://arxiv.org/abs/1905.10650) and propose a dynamic tuning mechanism that continuously regulates head activations during training, followed by post-training head pruning.

---

## Repository Structure

```
├── transtab/                         # Modified TransTab with head tuning support
│   ├── modeling_transtab.py          # Head gates, I_h computation
│   ├── dataset.py                    # Data loading from OpenML
│   ├── evaluator.py                  # Prediction and evaluation
│   └── ...
├── src/                              # Core modules (our contribution)
│   ├── importance.py                 # I_h computation and normalisation
│   ├── tuning.py                     # Gate freezing, updating from I_h
│   ├── dropping.py                   # Progressive head dropping evaluation
│   └── model_utils.py               # Model building and data helpers
├── scripts/
│   ├── train_vanilla.py              # Train with gates fixed at 1.0
│   ├── train_ih_tuned.py             # Train with dynamic I_h tuning
│   ├── evaluate_robustness.py        # Head dropping + AUC curves
│   ├── plot_auc_barplot.py           # AUC % change bar plot
│   ├── plot_heatmap.py               # I_h heatmap
│   └── plot_distribution.py          # Dataset distribution figures
├── data/
│   └── examples/                     # Example I_h scores (JSON)
├── checkpoints/                      # Saved model weights (git-ignored)
├── results/                          # Figures, CSVs, JSONs (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

conda create -n ih-tuning python=3.10 -y
conda activate ih-tuning
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train models

```bash
# Vanilla (gates fixed at 1.0)
python scripts/train_vanilla.py --dataset pc3

# I_h-tuned (dynamic gate updates after warmup)
python scripts/train_ih_tuned.py --dataset pc3 --warmup 10
```

### 2. Evaluate robustness to head pruning

```bash
# Full evaluation (loads checkpoints, runs all dropping strategies)
python scripts/evaluate_robustness.py --dataset pc3 --skip-training

# Regenerate plots from saved results
python scripts/evaluate_robustness.py --dataset pc3 --plot-only
```

### 3. Generate figures

```bash
# AUC % change bar plot
python scripts/plot_auc_barplot.py --font-size 16 --no-title

# I_h heatmap
python scripts/plot_heatmap.py --json data/examples/pc3_importance.json

# Dataset distribution
python scripts/plot_distribution.py --excel path/to/datasets.xlsx
```

---

## Common CLI Flags

All scripts share a consistent interface:

| Flag | Description |
|---|---|
| `--dataset NAME` | OpenML dataset name |
| `--font-size N` | Font size for all plot text |
| `--no-title` | Remove titles |
| `--no-legend` | Remove legends |
| `--dpi N` | Output resolution |

---

## Experimental Cases

| Case | I_h Tuning | Head Pruning Strategy |
|---|---|---|
| 1 | No | None (vanilla baseline) |
| 2 | Yes | None (I_h-tuned baseline) |
| 3 | No | Random |
| 4 | Yes | Random |
| 5 | No | Prune I_h min → max |
| 6 | Yes | Prune I_h min → max |
| 7 | No | Prune I_h max → min |
| 8 | Yes | Prune I_h max → min |

---

## Datasets

All 40 binary classification datasets are from [OpenML](https://www.openml.org/) and loaded automatically via `transtab.load_data()`.

---

## Citation

```bibtex
@inproceedings{yourname2025,
    title     = {Your Paper Title},
    author    = {Your Name and Others},
    booktitle = {Venue},
    year      = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
