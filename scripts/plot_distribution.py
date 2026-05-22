

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path


CFG = {

                :   "Results.xlsx",

              :     [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000],
                :   ["101-500", "501-1K", "1K-5K", "5K-10K", "10K-20K", "20K-50K", "50K-100K"],

                  :   [0, 10, 50, float("inf")],
                    : ["0~10 cols", "10~50", ">50"],

                :  ["#98D1CF", "#B99ADF"],                                  
                :  {                                                        
                   : "#BFE4E2",
               :     "#FFC37D",
             :       "#FA7567",
    },

                  :      (6, 6),                              
                  :      (13, 6),                             
                       : (18, 5.4),                                  
               :         0.62,
                   :     1.6,
                     :   1.8,
                 :       6,
                    :    35,                                                

                                                                          
               :    16,
                :   True,                                  
                 :  True,                                   

                                                               
         :          300,
}


def setup(cfg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    cfg["script_dir"]   = script_dir
    cfg["results_dir"]  = results_dir

    excel = cfg["excel_file"]
    if not os.path.isabs(excel):
        excel = os.path.join(script_dir, excel)
    cfg["excel_path"] = excel

                              
    fs = cfg["font_size"]
    plt.rcParams.update({
                    :          140,
                     :         cfg["dpi"],
                     :         "DejaVu Sans",
                   :           fs,
                        :      fs,
                        :      fs,
                         :     fs,
                         :     fs,
                         :     fs,
                         :     False,
                           :   False,
    })

    print(f"\n{'='*60}")
    print(f"  Excel file  : {cfg['excel_path']}")
    print(f"  Results dir : {results_dir}")
    print(f"  Font size   : {fs}")
    print(f"  Show title  : {cfg['show_title']}")
    print(f"  Show legend : {cfg['show_legend']}")
    print(f"{'='*60}\n")

    return cfg


def load_data(cfg):
                                        
    df = pd.read_excel(cfg["excel_path"]).iloc[:, :5].copy()
    df.columns = ["dataset", "n_features", "n_categorical", "n_numerical", "n_rows"]

    number_cols = ["n_features", "n_categorical", "n_numerical", "n_rows"]
    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dataset", "n_rows"])
    df[number_cols] = df[number_cols].fillna(0).astype(int)
    df = df[df["n_rows"] > 0].copy()

                           
    df["row_group"] = pd.cut(
        df["n_rows"],
        bins=cfg["row_bins"],
        labels=cfg["row_labels"],
        include_lowest=True,
        right=True,
    )
    df["feature_group"] = pd.cut(
        df["n_features"],
        bins=cfg["feature_bins"],
        labels=cfg["feature_labels"],
        include_lowest=True,
        right=True,
    )

                               
    dimension_counts = (
        df.groupby(["row_group", "feature_group"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=cfg["row_labels"], columns=cfg["feature_labels"], fill_value=0)
    )

    print(f"  Datasets loaded : {len(df)}")
    print(f"  Categorical cols: {df['n_categorical'].sum():,}")
    print(f"  Numerical cols  : {df['n_numerical'].sum():,}")
    print(f"  Rows range      : {df['n_rows'].min():,} – {df['n_rows'].max():,}")
    print(f"  Features range  : {df['n_features'].min():,} – {df['n_features'].max():,}\n")

    return df, dimension_counts


def _draw_pie(ax, df, cfg):
                                                 
    fs = cfg["font_size"]
    totals = [df["n_categorical"].sum(), df["n_numerical"].sum()]
    labels = ["Categorical/Textual", "Numerical"]

    wedges, _, autotexts = ax.pie(
        totals,
        colors=cfg["pie_colors"],
        startangle=90,
        counterclock=False,
        autopct="%1.2f%%",
        textprops={"fontsize": fs, "color": "black"},
    )
    ax.axis("equal")

    if cfg["show_legend"]:
        ax.legend(
            wedges, labels,
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.15),
            frameon=False,
            fontsize=fs,
        )
    if cfg["show_title"]:
        ax.set_title("Column Type Distribution", fontsize=fs, pad=45)


def _draw_bar(ax, dimension_counts, cfg):
                                             
    fs = cfg["font_size"]
    x      = np.arange(len(cfg["row_labels"]))
    bottom = np.zeros(len(cfg["row_labels"]))

    for label in cfg["feature_labels"]:
        values = dimension_counts[label].to_numpy()
        ax.bar(
            x, values,
            bottom=bottom,
            color=cfg["bar_colors"][label],
            edgecolor="black",
            linewidth=cfg["bar_linewidth"],
            width=cfg["bar_width"],
            label=label,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(
        cfg["row_labels"],
        fontsize=fs,
        fontweight="bold",
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlabel("Number of samples", fontsize=fs, fontweight="bold")
    ax.set_ylabel("Number of datasets", fontsize=fs, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fs, width=cfg["bar_linewidth"], length=cfg["tick_length"])
    ax.tick_params(axis="x", width=cfg["bar_linewidth"], length=cfg["tick_length"])
    ax.spines["left"].set_linewidth(cfg["spine_linewidth"])
    ax.spines["bottom"].set_linewidth(cfg["spine_linewidth"])

    if cfg["show_legend"]:
        handles = [
            Patch(
                facecolor=cfg["bar_colors"][label],
                edgecolor="black",
                linewidth=cfg["bar_linewidth"],
                label=label,
            )
            for label in cfg["feature_labels"]
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=fs)

    if cfg["show_title"]:
        ax.set_title("Dimensions of Tables", fontsize=fs, pad=12)


def plot_pie(df, cfg):
                               
    fig, ax = plt.subplots(figsize=cfg["fig_size_pie"])
    _draw_pie(ax, df, cfg)
    fig.tight_layout()
    path = os.path.join(cfg["results_dir"], "column_type_distribution.png")
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")


def plot_bar(dimension_counts, cfg):
                                       
    fig, ax = plt.subplots(figsize=cfg["fig_size_bar"])
    _draw_bar(ax, dimension_counts, cfg)
    fig.tight_layout()
    path = os.path.join(cfg["results_dir"], "dimensions_of_tables.png")
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")


def plot_combined(df, dimension_counts, cfg):
                                                             
    fig, (ax_pie, ax_bar) = plt.subplots(
        1, 2,
        figsize=cfg["fig_size_combined"],
        gridspec_kw={"width_ratios": [2, 6]},
    )
    _draw_pie(ax_pie, df, cfg)
    _draw_bar(ax_bar, dimension_counts, cfg)
    fig.tight_layout(w_pad=3.0)
    path = os.path.join(cfg["results_dir"], "dataset_distribution_combined.png")
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")


def generate_all_plots(df, dimension_counts, cfg):
    print("\n" + "─"*50)
    print("  Generating figures ...")
    print("─"*50)
    plot_pie(df, cfg)
    plot_bar(df if False else dimension_counts, cfg)                             
    plot_combined(df, dimension_counts, cfg)


def generate_all_plots(df, dimension_counts, cfg):
    print("\n" + "─"*50)
    print("  Generating figures ...")
    print("─"*50)
    plot_pie(df, cfg)
    plot_bar(dimension_counts, cfg)
    plot_combined(df, dimension_counts, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Distribution Figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--excel",      default=None, metavar="PATH",
                        help="Path to the Excel file (overrides CFG)")
    parser.add_argument("--font-size",  type=int,   default=None, metavar="N",
                        help="Font size for all text (default: 16)")
    parser.add_argument("--no-title",   action="store_true",
                        help="Remove titles from all figures")
    parser.add_argument("--no-legend",  action="store_true",
                        help="Remove legends from all figures")
    parser.add_argument("--dpi",        type=int,   default=None, metavar="N",
                        help="DPI for saved figures (default: 300)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.excel:
        CFG["excel_file"] = args.excel
    if args.font_size:
        CFG["font_size"] = args.font_size
    if args.no_title:
        CFG["show_title"] = False
    if args.no_legend:
        CFG["show_legend"] = False
    if args.dpi:
        CFG["dpi"] = args.dpi

    cfg = setup(CFG)

    df, dimension_counts = load_data(cfg)
    generate_all_plots(df, dimension_counts, cfg)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {cfg['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()