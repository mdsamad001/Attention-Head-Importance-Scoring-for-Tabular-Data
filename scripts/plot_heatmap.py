

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
except ImportError:
    print("seaborn not found — install with:  pip install seaborn")
    sys.exit(1)

CFG = {

               :    "head_importance_eye_movements.json",

               :    True,                          
              :     True,                               
          :         "RdYlGn",
                :   True,                                        
               :    ".3f",                                   
              :     (12, 4),                             
               :    11,
                :   False,                                           
         :          150,
}


def setup(cfg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    cfg["script_dir"]  = script_dir
    cfg["results_dir"] = results_dir

                       
    jf = cfg["json_file"]
    if not os.path.isabs(jf):
        jf = os.path.join(script_dir, jf)
    cfg["json_path"] = jf

                         
    fs = cfg["font_size"]
    plt.rcParams.update({
                    :       120,
                     :      cfg["dpi"],
                     :      "DejaVu Sans",
                   :        fs,
                        :   fs,
                        :   fs,
                         :  fs,
                         :  fs,
    })

    print(f"\n{'='*60}")
    print(f"  JSON file   : {jf}")
    print(f"  Results dir : {results_dir}")
    print(f"  Font size   : {fs}")
    print(f"  Show title  : {cfg['show_title']}")
    print(f"  Show annot  : {cfg['show_annot']}")
    print(f"  Colormap    : {cfg['cmap']}")
    print(f"{'='*60}\n")
    return cfg


def load_data(cfg):
    with open(cfg["json_path"]) as f:
        data = json.load(f)

    layers = data["layers"]
    n_layers = len(layers)
    n_heads  = len(next(iter(layers.values()))["raw"])

    raw_matrix  = np.zeros((n_layers, n_heads))
    norm_matrix = np.zeros((n_layers, n_heads))

    for li in range(n_layers):
        raw_matrix[li]  = layers[str(li)]["raw"]
        norm_matrix[li] = layers[str(li)]["norm"]

    cfg["n_layers"]  = n_layers
    cfg["n_heads"]   = n_heads
    cfg["dataset"]   = data.get("dataset", "")
    cfg["model"]     = data.get("model",   "")

    print(f"  Loaded  {n_layers} layers × {n_heads} heads"
          f"  (dataset={cfg['dataset']}, model={cfg['model']})\n")

    return raw_matrix, norm_matrix


def _draw_heatmap(matrix, cfg, score_type):
    

    fs       = cfg["font_size"]
    n_layers = cfg["n_layers"]
    n_heads  = cfg["n_heads"]

    cbar_label = (
                          
        if score_type == "norm"
        else "Raw importance (I_h)"
    )

    fig, ax = plt.subplots(figsize=cfg["fig_size"])

    hm = sns.heatmap(
        matrix,
        ax=ax,
        annot=cfg["show_annot"],
        fmt=cfg["annot_fmt"],
        annot_kws={"size": fs},
        cmap=cfg["cmap"],
        vmin=0.1,                                      
        vmax=0.9,                                      
        xticklabels=[f"Head {i}" for i in range(n_heads)],
        yticklabels=[f"Layer {i}" for i in range(n_layers)],
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.95},
        linewidths=0.4,
        linecolor="#cccccc",
    )

                        
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(cbar_label, fontsize=fs)

    ax.set_xlabel("Attention head", fontsize=fs, fontweight="bold")
    ax.set_ylabel("Transformer layer", fontsize=fs, fontweight="bold")
    ax.tick_params(axis="x", labelsize=fs, rotation=0)
    ax.tick_params(axis="y", labelsize=fs, rotation=0)

    if cfg["show_title"]:
        suffix = "L2-Normalised" if score_type == "norm" else "Raw"
        ax.set_title(
            f"Head Importance ({suffix}) — {cfg['dataset']}",
            fontsize=fs, fontweight="bold", pad=10,
        )

    fig.tight_layout()

    ds   = cfg["dataset"].replace(" ", "_") if cfg["dataset"] else "unknown"
    name = f"head_importance_{ds}_{score_type}.png"
    path = os.path.join(cfg["results_dir"], name)
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  →  {path}")


def generate_all_plots(raw_matrix, norm_matrix, cfg):
    print("\n" + "─"*50)
    print("  Generating figures ...")
    print("─"*50)
    if cfg["plot_norm"]:
        _draw_heatmap(norm_matrix, cfg, score_type="norm")
    if cfg["plot_raw"]:
        _draw_heatmap(raw_matrix,  cfg, score_type="raw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head Importance Heatmap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json",       default=None, metavar="PATH",
                        help="Path to the JSON importance file")
    parser.add_argument("--font-size",  type=int, default=None, metavar="N",
                        help="Font size for all text (default: 11)")
    parser.add_argument("--no-title",   action="store_true",
                        help="Remove figure titles")
    parser.add_argument("--no-annot",   action="store_true",
                        help="Hide cell value annotations")
    parser.add_argument("--cmap",       default=None, metavar="NAME",
                        help="Matplotlib colormap name (default: RdYlGn)")
    parser.add_argument("--norm-only",  action="store_true",
                        help="Only generate the normalised heatmap")
    parser.add_argument("--raw-only",   action="store_true",
                        help="Only generate the raw heatmap")
    parser.add_argument("--dpi",        type=int, default=None, metavar="N",
                        help="DPI for saved figures (default: 150)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.json:
        CFG["json_file"]  = args.json
    if args.font_size:
        CFG["font_size"]  = args.font_size
    if args.no_title:
        CFG["show_title"] = False
    if args.no_annot:
        CFG["show_annot"] = False
    if args.cmap:
        CFG["cmap"]       = args.cmap
    if args.norm_only:
        CFG["plot_raw"]   = False
    if args.raw_only:
        CFG["plot_norm"]  = False
    if args.dpi:
        CFG["dpi"]        = args.dpi

    cfg = setup(CFG)
    raw_matrix, norm_matrix = load_data(cfg)
    generate_all_plots(raw_matrix, norm_matrix, cfg)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {cfg['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()