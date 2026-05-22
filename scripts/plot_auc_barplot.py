
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


CFG = {
              :     "all_csv_files",
                 :    [10, 20, 30, 40],
                  :   "auc_pct_change_vs_vanilla_baseline",
             : [
        {"key": "ih_driven_ih_order",    "filename": "ih_driven_ih_order.csv",
                : "Ih tuned - drop Ih_min to Ih_max"},
        {"key": "ih_driven_random",      "filename": "ih_driven_random.csv",
                : "Ih tuned - random dropping"},
        {"key": "ih_driven_worst_first", "filename": "ih_driven_worst_first.csv",
                : "Ih tuned - drop Ih_max to Ih_min"},
        {"key": "vanilla_ih_order",      "filename": "vanilla_ih_order.csv",
                : "Vanilla - drop Ih_min to Ih_max"},
        {"key": "vanilla_random",        "filename": "vanilla_random.csv",
                : "Vanilla - random dropping"},
        {"key": "vanilla_worst_first",   "filename": "vanilla_worst_first.csv",
                : "Vanilla - drop Ih_max to Ih_min"},
    ],
                :     ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"],
                    : {
                            :    "#4E79A7",
                          :      "#59A14F",
                               : "#F28E2B",
                          :      "#E15759",
                        :        "#76B7B2",
                             :   "#EDC948",
    },

                   :    (13.5, 7.2),   
                      : (13.0, 5.5),   
                    :   0.72,
                       : 0.13,


               :    12,
                :   True,   
                 :  True,   

         : 300,
}


def setup(cfg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    cfg["script_dir"]  = script_dir
    cfg["results_dir"] = results_dir

    root = cfg["csv_root"]
    if not os.path.isabs(root):
        root = os.path.join(script_dir, root)
    cfg["csv_root_path"] = root

    fs = cfg["font_size"]
    plt.rcParams.update({
                    :       140,
                     :      cfg["dpi"],
                     :      "DejaVu Sans",
                   :        fs,
                        :   fs,
                        :   fs,
                         :  fs,
                         :  fs,
                         :  fs,
                         :  False,
                           : False,
    })

    print(f"\n{'='*60}")
    print(f"  CSV root    : {root}")
    print(f"  Results dir : {results_dir}")
    print(f"  Font size   : {fs}")
    print(f"  Show title  : {cfg['show_title']}")
    print(f"  Show legend : {cfg['show_legend']}")
    print(f"{'='*60}\n")
    return cfg

def _load_one(dataset_dir, method, cfg):
                                                           
    csv_path = Path(dataset_dir) / method["filename"]
    dataset_name = Path(dataset_dir).name

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing CSV for dataset '{dataset_name}': {csv_path}"
        )

    df = pd.read_csv(csv_path)
    required = {"dataset", "model", "strategy", "num_dropped", cfg["value_column"]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns in '{csv_path}': {missing}"
        )

    df["num_dropped"]        = pd.to_numeric(df["num_dropped"],        errors="raise").astype(int)
    df[cfg["value_column"]]  = pd.to_numeric(df[cfg["value_column"]],  errors="raise").astype(float)

    records = []
    for drop in cfg["drop_levels"]:
        rows = df.loc[df["num_dropped"] == drop]
        if len(rows) != 1:
            raise ValueError(
                f"Expected 1 row for num_dropped={drop}, found {len(rows)} "
                f"in '{csv_path}'"
            )
        records.append({
                     :     dataset_name,
                        :  method["key"],
                         : drop,
                            : float(rows.iloc[0][cfg["value_column"]]),
        })
    return records


def load_data(cfg):
                                                       
    root = cfg["csv_root_path"]
    if not os.path.exists(root):
        raise FileNotFoundError(f"CSV root not found: {root}")

    dataset_dirs = sorted(
        [p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset folders found under: {root}")

    all_records = []
    for d in dataset_dirs:
        for m in cfg["methods"]:
            all_records.extend(_load_one(d, m, cfg))

    raw_df = pd.DataFrame(all_records)
    method_order  = [m["key"] for m in cfg["methods"]]
    label_lookup  = {m["key"]: m["label"] for m in cfg["methods"]}

    summary_df = (
        raw_df
        .groupby(["method_key", "num_dropped"], as_index=False)
        .agg(
            mean_pct_change=("auc_pct_change", "mean"),
            std_pct_change= ("auc_pct_change", "std"),
            n_datasets=     ("dataset",        "nunique"),
        )
    )
    summary_df["std_pct_change"] = summary_df["std_pct_change"].fillna(0.0)
    summary_df["se_pct_change"]  = (
        summary_df["std_pct_change"] / np.sqrt(summary_df["n_datasets"])
    )
    summary_df["method_label"] = summary_df["method_key"].map(label_lookup)
    summary_df["method_key"]   = pd.Categorical(
        summary_df["method_key"], categories=method_order, ordered=True
    )
    summary_df = summary_df.sort_values(["method_key", "num_dropped"]).reset_index(drop=True)

    print(f"  Datasets loaded : {len(dataset_dirs)}"
          f"  ({[d.name for d in dataset_dirs]})")
    print(f"  Strategies      : {len(cfg['methods'])}")
    print(f"  Drop levels     : {cfg['drop_levels']}\n")

    return raw_df, summary_df


def _y_limits(summary_df):
    y_min  = float((summary_df["mean_pct_change"] - summary_df["se_pct_change"]).min())
    y_max  = max(0.0, float((summary_df["mean_pct_change"] + summary_df["se_pct_change"]).max()))
    y_span = y_max - y_min
    y_pad  = 1.0 if math.isclose(y_span, 0.0) else y_span * 0.12
    return y_min, y_max, y_pad


def _style_ax(ax):
                                    
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_grid(summary_df, cfg):
                                                       
    fs          = cfg["font_size"]
    drop_levels = cfg["drop_levels"]
    x_pos       = np.arange(len(drop_levels))
    y_min, y_max, y_pad = _y_limits(summary_df)

    fig, axes = plt.subplots(2, 3, figsize=cfg["fig_size_grid"], sharey=True)
    axes = axes.ravel()

    for ax, method in zip(axes, cfg["methods"]):
        msub = (
            summary_df.loc[summary_df["method_key"] == method["key"]]
            .set_index("num_dropped")
            .loc[drop_levels]
            .reset_index()
        )

        ax.bar(
            x_pos,
            msub["mean_pct_change"],
            yerr=msub["se_pct_change"],
            capsize=4,
            color=cfg["bar_colors"],
            edgecolor="#333333",
            linewidth=0.8,
            width=cfg["bar_width_grid"],
            error_kw={"elinewidth": 1.1, "ecolor": "#222222"},
        )

        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(d) for d in drop_levels], fontsize=fs)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        _style_ax(ax)

        n_vals = msub["n_datasets"].unique()
        n_text = f"n={int(n_vals[0])}" if len(n_vals) == 1 else "n varies"
        ax.text(0.98, 0.04, n_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs, color="#333333")

        if cfg["show_title"]:
            ax.set_title(method["label"], fontweight="bold", fontsize=fs)

                        
    fig.supxlabel("Number of heads dropped", fontsize=fs, y=0.02)
    fig.supylabel("Mean % change in AUC", fontsize=fs, x=0.01)

    if cfg["show_title"]:
        fig.suptitle(
                                                                             ,
            fontsize=fs, fontweight="bold", y=0.99,
        )

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.95])
    path = os.path.join(cfg["results_dir"], "auc_pct_change_grid.png")
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")


def plot_grouped(summary_df, cfg):
                                                            
    fs          = cfg["font_size"]
    drop_levels = cfg["drop_levels"]
    methods     = cfg["methods"]

    x        = np.arange(len(drop_levels), dtype=float)
    n_m      = len(methods)
    offsets  = (np.arange(n_m) - (n_m - 1) / 2) * cfg["bar_width_grouped"]

    y_min, y_max, y_pad = _y_limits(summary_df)

    fig, ax = plt.subplots(figsize=cfg["fig_size_grouped"])

    for idx, method in enumerate(methods):
        msub = (
            summary_df.loc[summary_df["method_key"].astype(str) == method["key"]]
            .set_index("num_dropped")
            .loc[drop_levels]
            .reset_index()
        )
        means  = msub["mean_pct_change"].to_numpy(dtype=float)
        errors = msub["se_pct_change"].to_numpy(dtype=float)

        ax.bar(
            x + offsets[idx],
            means,
            width=cfg["bar_width_grouped"],
            yerr=errors,
            capsize=2.5,
            label=method["label"],
            color=cfg["grouped_colors"][method["key"]],
            edgecolor="#222222",
            linewidth=0.6,
            error_kw={"elinewidth": 0.9, "ecolor": "#222222", "capthick": 0.9},
            zorder=3,
        )

    ax.axhline(0, color="#111111", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in drop_levels], fontsize=fs)
    ax.set_xlabel("Number of heads dropped", fontsize=fs, fontweight="bold")
    ax.set_ylabel("Mean % change in AUC", fontsize=fs, fontweight="bold")
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 1.5)
    _style_ax(ax)

    if cfg["show_title"]:
        ax.set_title(
                                                                             ,
            fontsize=fs, fontweight="bold", pad=10,
        )

    if cfg["show_legend"]:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.85),                               
            ncol=2,                                                    
            frameon=False,
            fontsize=18,
        )

                              
    n_vals = summary_df["n_datasets"].unique()
    n_text = f"n={int(n_vals[0])}" if len(n_vals) == 1 else "n varies"
    ax.text(0.99, 0.04, n_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fs, color="#333333")

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)                                        
    path = os.path.join(cfg["results_dir"], "auc_pct_change_grouped.png")
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")

def generate_all_plots(raw_df, summary_df, cfg):
    print("\n" + "─"*50)
    print("  Generating figures ...")
    print("─"*50)
    plot_grid(summary_df, cfg)
    plot_grouped(summary_df, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AUC Percentage Change Bar Plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-root",   default=None, metavar="PATH",
                        help="Root folder containing per-dataset CSV subfolders")
    parser.add_argument("--font-size",  type=int, default=None, metavar="N",
                        help="Font size for all text (default: 12)")
    parser.add_argument("--no-title",   action="store_true",
                        help="Remove all titles and suptitles")
    parser.add_argument("--no-legend",  action="store_true",
                        help="Remove legend from grouped bar chart")
    parser.add_argument("--dpi",        type=int, default=None, metavar="N",
                        help="DPI for saved figures (default: 300)")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.csv_root:
        CFG["csv_root"]   = args.csv_root
    if args.font_size:
        CFG["font_size"]  = args.font_size
    if args.no_title:
        CFG["show_title"] = False
    if args.no_legend:
        CFG["show_legend"] = False
    if args.dpi:
        CFG["dpi"] = args.dpi

    cfg = setup(CFG)

    raw_df, summary_df = load_data(cfg)
    generate_all_plots(raw_df, summary_df, cfg)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {cfg['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()