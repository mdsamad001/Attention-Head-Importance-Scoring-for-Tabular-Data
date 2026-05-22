

import os
import gc
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from copy import deepcopy
from tqdm import tqdm


CFG = {
                                                               
                                                      
                   :   "/home/cidalab/Ahmad/Transtab_Ih",
             :         "eucalyptus",

                                                               
                :   128,
             :      256,
                :   6,
               :    8,
             :      0.1,

                                                               
                  :   50,
                   :  10,
                :     64,
                   :  32,
        :             1e-4,
                  :   1e-5,
              :       10,
          :           42,

                                                                
               :       None,
                                                                  
               :       None,
                                                                
                   :  True,
                                           
              :       (16, 8),
                           
             :        150,
                                                                 
               :      16,
                                         
                 :    True,
}


def setup(cfg):
                                             
                                                               
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)

    ckpt_dir    = os.path.join(repo_root, "checkpoints")
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cfg["script_dir"]  = script_dir
    cfg["repo_root"]   = repo_root
    cfg["ckpt_dir"]    = ckpt_dir
    cfg["out_dir"]     = results_dir

    ds = cfg["dataset"]

                                                                      
    cfg["ckpt_a"]       = os.path.join(ckpt_dir, f"{ds}_vanilla.pt")
    cfg["ckpt_b"]       = os.path.join(ckpt_dir, f"{ds}_ih_tuned.pt")

                                                        
    cfg["results_json"] = os.path.join(results_dir, f"{ds}_results.json")
    cfg["results_csv"]  = os.path.join(results_dir, f"{ds}_auc_curves.csv")
    cfg["summary_csv"]  = os.path.join(results_dir, f"{ds}_summary.csv")

                          
    if cfg["transtab_path"] not in sys.path:
        sys.path.insert(0, cfg["transtab_path"])
    os.chdir(cfg["transtab_path"])

            
    cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

           
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    print(f"\n{'='*60}")
    print(f"  Dataset     : {cfg['dataset']}")
    print(f"  Device      : {cfg['device']}")
    print(f"  Repo root   : {repo_root}")
    print(f"  Checkpoints : {ckpt_dir}")
    print(f"  Results     : {results_dir}")
    print(f"{'='*60}\n")

    return cfg


def load_dataset(cfg):
    import transtab
    transtab.random_seed(cfg["seed"])

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols =        transtab.load_data(cfg["dataset"])

    X_train, y_train = trainset
    num_class   = len(np.unique(y_train.values if hasattr(y_train, "values") else y_train))

    cfg["cat_cols"]     = cat_cols
    cfg["num_cols"]     = num_cols
    cfg["bin_cols"]     = bin_cols
    cfg["num_class"]    = num_class
    cfg["eval_metric"]  = "auc"               
    cfg["total_heads"]  = cfg["num_layers"] * cfg["num_heads"]

    print(f"  Train={len(X_train)}  Val={len(valset[0])}  Test={len(testset[0])}")
    print(f"  Classes={num_class}  Metric=auc\n")

    return trainset, valset, testset


def get_device(model):
    return next(model.parameters()).device


def build_model(cfg):
    import transtab
    transtab.random_seed(cfg["seed"])
    return transtab.build_classifier(
        categorical_columns=cfg["cat_cols"],
        numerical_columns=cfg["num_cols"],
        binary_columns=cfg["bin_cols"],
        num_class=cfg["num_class"],
        hidden_dim=cfg["hidden_dim"],
        num_layer=cfg["num_layers"],
        num_attention_head=cfg["num_heads"],
        hidden_dropout_prob=cfg["dropout"],
        ffn_dim=cfg["ffn_dim"],
        device=cfg["device"],
    )


def set_all_gates(model, cfg, value=1.0):
                                                        
    gates = torch.full((cfg["num_heads"],), value, device=get_device(model))
    for li in range(cfg["num_layers"]):
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(gates.clone())


def freeze_gate_logits(model):
    for layer in model.encoder.transformer_encoder:
        if hasattr(layer.self_attn, "head_gate_logits"):
            layer.self_attn.head_gate_logits.requires_grad_(False)


def compute_importance_unbiased(model, dataset, cfg):
    

    X, y = dataset
    n_layers = model.encoder.num_layer

                                   
    saved = {}
    for li in range(n_layers):
        attn = model.encoder.transformer_encoder[li].self_attn
        saved[li] = attn.head_gate_logits.data.clone()
        attn.head_gate_logits.data.fill_(10.0)                     

    model.eval()
    model.enable_head_importance_tracking(True)

    accumulated = None
    n_samples   = 0
    indices     = np.arange(len(X))
    bs          = cfg["ih_batch_size"]
    n_batches   = (len(X) + bs - 1) // bs

    for b in tqdm(range(n_batches), desc="  Computing I_h", leave=False):
        start = b * bs
        end   = min(start + bs, len(X))
        bx, by = X.iloc[indices[start:end]], y.iloc[indices[start:end]]

        model.zero_grad()
        _, loss = model(bx, by)
        loss.mean().backward()

        batch_imp = model.compute_head_importance()
        if accumulated is None:
            accumulated = {k: v.clone() for k, v in batch_imp.items()}
        else:
            for k, v in batch_imp.items():
                accumulated[k] += v
        n_samples += (end - start)

    model.enable_head_importance_tracking(False)
    for k in accumulated:
        accumulated[k] /= n_samples

             
    for li in range(n_layers):
        model.encoder.transformer_encoder[li].self_attn.head_gate_logits.data.copy_(saved[li])

    return accumulated


def l2_normalize(importance_dict):
    return {
        li: scores / (torch.norm(scores, p=2) + 1e-8)
        for li, scores in importance_dict.items()
    }


def set_xi_from_importance(model, importance_dict):
                                                        
    normed = l2_normalize(importance_dict)
    for li, scores in normed.items():
        clamped = torch.clamp(scores, 1e-6, 1 - 1e-6)
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(clamped)
    return normed


def _save_restore_masks(model, cfg):
                                                               
    saved = {}
    for li in range(cfg["num_layers"]):
        saved[li] = model.encoder.transformer_encoder[li].self_attn.get_head_masks().clone()
    return saved


def _restore_masks(model, saved):
    for li, mask in saved.items():
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(mask)


def _evaluate(model, X_test, y_test, metric):
    import transtab
    preds = transtab.predict(model, X_test)
    return transtab.evaluate(preds, y_test, metric=metric)[0]


def progressive_head_dropping(model, testset, importance, cfg):
    

    import transtab
    X_test, y_test = testset
    metric = cfg["eval_metric"]

    all_heads = [
        {"layer": li, "head": hi, "importance": float(score)}
        for li, scores in importance.items()
        for hi, score in enumerate(scores.cpu().numpy() if torch.is_tensor(scores) else scores)
    ]
    sorted_heads = sorted(all_heads, key=lambda h: h["importance"])

    saved = _save_restore_masks(model, cfg)
    baseline = _evaluate(model, X_test, y_test, metric)
    auc_values = [baseline]

    for h in tqdm(sorted_heads, desc="  I_h-order drop", leave=False):
        li, hi = h["layer"], h["head"]
        mask = model.encoder.transformer_encoder[li].self_attn.get_head_masks().clone()
        mask[hi] = 0.0
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(mask)
        auc_values.append(_evaluate(model, X_test, y_test, metric))

    _restore_masks(model, saved)
    return auc_values, sorted_heads


def drop_by_order(model, testset, heads_order, cfg):
                                                                 
    X_test, y_test = testset
    metric = cfg["eval_metric"]

    saved    = _save_restore_masks(model, cfg)
    baseline = _evaluate(model, X_test, y_test, metric)
    auc_values = [baseline]

    for h in tqdm(heads_order, desc="  Custom-order drop", leave=False):
        li, hi = h["layer"], h["head"]
        mask = model.encoder.transformer_encoder[li].self_attn.get_head_masks().clone()
        mask[hi] = 0.0
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(mask)
        auc_values.append(_evaluate(model, X_test, y_test, metric))

    _restore_masks(model, saved)
    return auc_values


def train_vanilla(cfg, trainset, valset):
                                                            
    import transtab

    print("\n" + "─"*50)
    print("  Training Model A — Vanilla (gates fixed at 1.0)")
    print("─"*50)

    model = build_model(cfg)
    freeze_gate_logits(model)
    set_all_gates(model, cfg, value=1.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    X_train, y_train = trainset
    X_val,   y_val   = valset
    metric           = cfg["eval_metric"]
    history          = {"epoch": [], "train_loss": [], "val_metric": []}
    best_metric      = -float("inf")
    best_state       = None
    patience_ctr     = 0

    for epoch in range(cfg["total_epochs"]):
        model.train()
        indices   = np.random.permutation(len(X_train))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(X_train), cfg["batch_size"]):
            idx = indices[i : i + cfg["batch_size"]]
            bx, by = X_train.iloc[idx], y_train.iloc[idx]
            optimizer.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        model.eval()
        val_metric = _evaluate(model, X_val, y_val, metric)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_metric"].append(val_metric)

        if val_metric > best_metric:
            best_metric  = val_metric
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  val_{metric}={val_metric:.4f}"
                  f"  patience={patience_ctr}/{cfg['patience']}")

        if patience_ctr >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(cfg["device"])
    torch.save(best_state, cfg["ckpt_a"])

    test_metric = _evaluate(model, *cfg["_testset"], metric)
    print(f"  ✓ Saved  →  {cfg['ckpt_a']}")
    print(f"  Best val_{metric}={best_metric:.4f}   test_{metric}={test_metric:.4f}")

    return model, history, best_metric, test_metric


def train_ih_driven(cfg, trainset, valset):
                                                             
    import transtab

    print("\n" + "─"*50)
    print("  Training Model B — I_h-Driven")
    print("─"*50)

    model = build_model(cfg)
    freeze_gate_logits(model)
    set_all_gates(model, cfg, value=1.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    X_train, y_train = trainset
    X_val,   y_val   = valset
    metric           = cfg["eval_metric"]
    history          = {"epoch": [], "train_loss": [], "val_metric": []}
    best_metric      = -float("inf")
    best_state       = None
    patience_ctr     = 0

    for epoch in range(cfg["total_epochs"]):
        model.train()
        indices   = np.random.permutation(len(X_train))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(X_train), cfg["batch_size"]):
            idx = indices[i : i + cfg["batch_size"]]
            bx, by = X_train.iloc[idx], y_train.iloc[idx]
            optimizer.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        model.eval()

                                                  
        if epoch >= cfg["warmup_epochs"]:
            imp = compute_importance_unbiased(model, trainset, cfg)
            set_xi_from_importance(model, imp)

        val_metric = _evaluate(model, X_val, y_val, metric)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_metric"].append(val_metric)

        if val_metric > best_metric:
            best_metric  = val_metric
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        phase = "WARMUP" if epoch < cfg["warmup_epochs"] else "I_h→ξ"
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg["warmup_epochs"]:
            print(f"  Epoch {epoch:3d} [{phase}]  loss={avg_loss:.4f}  "
                  f"val_{metric}={val_metric:.4f}  patience={patience_ctr}/{cfg['patience']}")

        if patience_ctr >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(cfg["device"])
    freeze_gate_logits(model)
    torch.save(best_state, cfg["ckpt_b"])

    test_metric = _evaluate(model, *cfg["_testset"], metric)
    print(f"  ✓ Saved  →  {cfg['ckpt_b']}")
    print(f"  Best val_{metric}={best_metric:.4f}   test_{metric}={test_metric:.4f}")

    return model, history, best_metric, test_metric


def load_checkpoints(cfg):
                                                      
    import transtab

    print("\n  Loading checkpoints (skip training)...")

    model_a = build_model(cfg)
    model_a.load_state_dict(torch.load(cfg["ckpt_a"], map_location=cfg["device"]))
    freeze_gate_logits(model_a)
    set_all_gates(model_a, cfg, value=1.0)
    print(f"  ✓ Model A loaded from {cfg['ckpt_a']}")

    model_b = build_model(cfg)
    model_b.load_state_dict(torch.load(cfg["ckpt_b"], map_location=cfg["device"]))
    freeze_gate_logits(model_b)
    print(f"  ✓ Model B loaded from {cfg['ckpt_b']}")

    return model_a, model_b


def evaluate_model(model, label, testset, cfg):
                                                              
    print(f"\n  Evaluating {label} ...")

                                
    print("    → Computing I_h ...")
    importance  = compute_importance_unbiased(model, testset, cfg)

    print("    → I_h order (lowest first) ...")
    auc_ih, sorted_heads = progressive_head_dropping(model, testset, importance, cfg)

                  
    random_heads = sorted_heads.copy()
    random.seed(cfg["seed"])
    random.shuffle(random_heads)
    print("    → Random order ...")
    auc_random   = drop_by_order(model, testset, random_heads, cfg)

                                     
    reverse_heads = sorted_heads[::-1]
    print("    → Worst-first order ...")
    auc_worst    = drop_by_order(model, testset, reverse_heads, cfg)

    return {
                  :     auc_ih[0],
                :       auc_ih,
                    :   auc_random,
                   :    auc_worst,
                      : sorted_heads,                                      
                    :   {
            str(li): scores.detach().cpu().tolist()
            for li, scores in importance.items()
        },
    }


def save_results(cfg, results, history_a, history_b,
                 best_val_a, best_val_b, test_metric_a, test_metric_b):

                                                               
    payload = {
                : {
            k: v for k, v in cfg.items()
            if isinstance(v, (str, int, float, bool, list, type(None)))
            and k not in ("cat_cols", "num_cols", "bin_cols", "_testset")
        },
                 : {
                   :          "Vanilla",
                      :       best_val_a,
                         :    test_metric_a,
                      :       history_a,
                      :       results["a"],
        },
                 : {
                   :          "I_h-Driven",
                      :       best_val_b,
                         :    test_metric_b,
                      :       history_b,
                      :       results["b"],
        },
    }

    with open(cfg["results_json"], "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ JSON saved  →  {cfg['results_json']}")

                                                                          
    n = len(results["a"]["auc_ih"])
    total_heads = cfg["total_heads"]
    rows = []
    for i in range(n):
        pct = round((i / total_heads) * 100, 2)
        rows.append({
                           :     i,
                               : pct,
                        :        results["a"]["auc_ih"][i]      if i < len(results["a"]["auc_ih"])     else None,
                            :    results["a"]["auc_random"][i]  if i < len(results["a"]["auc_random"]) else None,
                           :     results["a"]["auc_worst"][i]   if i < len(results["a"]["auc_worst"])  else None,
                          :      results["b"]["auc_ih"][i]      if i < len(results["b"]["auc_ih"])     else None,
                              :  results["b"]["auc_random"][i]  if i < len(results["b"]["auc_random"]) else None,
                             :   results["b"]["auc_worst"][i]   if i < len(results["b"]["auc_worst"])  else None,
        })

    df_curves = pd.DataFrame(rows)
    df_curves.to_csv(cfg["results_csv"], index=False)
    print(f"  ✓ AUC curves CSV saved  →  {cfg['results_csv']}")

                                                               
    metric = cfg["eval_metric"].upper()
    summary_rows = [
        {
                   :            "Vanilla",
            f"baseline_{metric}":  results["a"]["baseline"],
            f"test_{metric}":      test_metric_a,
            f"best_val_{metric}":  best_val_a,
                          :        results["a"]["auc_ih"][-1],
                              :    results["a"]["auc_random"][-1],
                             :     results["a"]["auc_worst"][-1],
        },
        {
                   :            "I_h-Driven",
            f"baseline_{metric}":  results["b"]["baseline"],
            f"test_{metric}":      test_metric_b,
            f"best_val_{metric}":  best_val_b,
                          :        results["b"]["auc_ih"][-1],
                              :    results["b"]["auc_random"][-1],
                             :     results["b"]["auc_worst"][-1],
        },
    ]
    pd.DataFrame(summary_rows).to_csv(cfg["summary_csv"], index=False)
    print(f"  ✓ Summary CSV saved     →  {cfg['summary_csv']}")


def load_results(cfg):
                                                                                             
    with open(cfg["results_json"]) as f:
        payload = json.load(f)

                                                    
    saved_cfg = payload.get("config", {})
    for key in ("eval_metric", "total_heads", "num_layers", "num_heads",
                         , "warmup_epochs"):
        if key in saved_cfg and key not in cfg:
            cfg[key] = saved_cfg[key]

                                                   
    if "total_heads" not in cfg:
        cfg["total_heads"] = cfg.get("num_layers", 6) * cfg.get("num_heads", 8)
    cfg["eval_metric"] = "auc"               

    r_a = payload["model_a"]["dropping"]
    r_b = payload["model_b"]["dropping"]

                                                           
    history_a  = payload["model_a"].get("training", {"epoch": [], "train_loss": [], "val_metric": []})
    history_b  = payload["model_b"].get("training", {"epoch": [], "train_loss": [], "val_metric": []})
    best_val_a = payload["model_a"]["best_val"]
    best_val_b = payload["model_b"]["best_val"]
    test_a     = payload["model_a"]["test_metric"]
    test_b     = payload["model_b"]["test_metric"]

    print(f"  ✓ Results loaded from  →  {cfg['results_json']}")
    print(f"  ✓ eval_metric={cfg['eval_metric']}  total_heads={cfg['total_heads']}")
    return {"a": r_a, "b": r_b}, history_a, history_b, best_val_a, best_val_b, test_a, test_b


BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"

def _x_axis(n_points, cfg):
                                                          
    total = cfg["total_heads"]
    if cfg["plot_x_as_pct"]:
                                                                        
                                                                 
        denom = max(total, n_points - 1, 1)
        xs    = [(i / denom) * 100 for i in range(n_points)]
        label = "Heads dropped (%)"
    else:
        xs    = list(range(n_points))
        label = "Number of Heads dropped"
    return xs, label


def _apply_axis_cfg(ax, cfg, y_label):
                                                              
    fs = cfg.get("font_size", 16)
    if cfg["plot_ylim"] is not None:
        ax.set_ylim(cfg["plot_ylim"])

    if cfg.get("plot_x_as_pct", True):
                                                                              
        x_min = cfg["plot_xlim"][0] if cfg["plot_xlim"] else 0
        x_max = cfg["plot_xlim"][1] if cfg["plot_xlim"] else 99
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(range(0, 101, 10))                                                 
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
    else:
        if cfg["plot_xlim"] is not None:
            ax.set_xlim(cfg["plot_xlim"])

    ax.set_ylabel(y_label, fontsize=fs, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=fs)


def plot_all_strategies(results, cfg):
                                                             
    metric = cfg["eval_metric"].upper()
    ra, rb = results["a"], results["b"]

    total = cfg["total_heads"]
    max_h = max(1, round(total * 0.99))                                       
    a_ih     = ra["auc_ih"][:max_h]
    a_random = ra["auc_random"][:max_h]
    a_worst  = ra["auc_worst"][:max_h]
    b_ih     = rb["auc_ih"][:max_h]
    b_random = rb["auc_random"][:max_h]
    b_worst  = rb["auc_worst"][:max_h]

    xs, x_label = _x_axis(max_h, cfg)

    fig, ax = plt.subplots(figsize=cfg["fig_size"])

               
    ax.axhline(y=ra["baseline"], color=BLUE,   ls="--", lw=2, alpha=0.65,
               label=f"Vanilla baseline")
    ax.axhline(y=rb["baseline"], color=ORANGE, ls="--", lw=2, alpha=0.65,
               label=f"Ih tuned baseline")

             
    ax.plot(xs, a_ih,     "o-",  color=BLUE,   lw=2.5, ms=5, alpha=0.9, label="Vanilla - drop Ih_min to Ih_max")
    ax.plot(xs, a_random, "s--", color=BLUE,   lw=2.5, ms=5, alpha=0.9, label="Vanilla - random dropping")
    ax.plot(xs, a_worst,  "^:",  color=BLUE,   lw=2.5, ms=5, alpha=0.9, label="Vanilla - drop Ih_max to Ih_min")

                
    ax.plot(xs, b_ih,     "o-",  color=ORANGE, lw=2.5, ms=5, alpha=0.9, label="Ih tuned - drop Ih_min to Ih_max")
    ax.plot(xs, b_random, "s--", color=ORANGE, lw=2.5, ms=5, alpha=0.9, label="Ih tuned - random dropping")
    ax.plot(xs, b_worst,  "^:",  color=ORANGE, lw=2.5, ms=5, alpha=0.9, label="Ih tuned - drop Ih_max to Ih_min")

    fs = cfg.get("font_size", 16)
    ax.set_xlabel(x_label, fontsize=fs, fontweight="bold")
    ax.set_title(''
    )
    if cfg.get("show_legend", True):
        ax.legend(fontsize=fs, loc="lower left", ncol=2, framealpha=0.95)
    _apply_axis_cfg(ax, cfg, f"{metric}" if cfg["plot_x_as_pct"] else metric)

    if cfg["plot_ylim"] is None:
        all_vals = a_ih + a_random + a_worst + b_ih + b_random + b_worst
        ax.set_ylim(min(all_vals) - 0.02, max(ra["baseline"], rb["baseline"]) + 0.02)

    plt.tight_layout()
    path = os.path.join(cfg["out_dir"], f"{cfg['dataset']}_all_strategies.png")
    fig.savefig(path, dpi=cfg["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure saved  →  {path}")


def plot_ih_order_only(results, cfg):
                                                                          
    metric = cfg["eval_metric"].upper()
    ra, rb = results["a"], results["b"]

    total = cfg["total_heads"]
    max_h = max(1, round(total * 0.99))               
    a_ih  = ra["auc_ih"][:max_h]
    b_ih  = rb["auc_ih"][:max_h]
    xs, x_label = _x_axis(max_h, cfg)

    fig, ax = plt.subplots(figsize=cfg["fig_size"])

    ax.axhline(y=ra["baseline"], color=BLUE,   ls="--", lw=2, alpha=0.65,
               label=f"Vanilla baseline ({ra['baseline']:.4f})")
    ax.axhline(y=rb["baseline"], color=ORANGE, ls="--", lw=2, alpha=0.65,
               label=f"Ih tuned baseline ({rb['baseline']:.4f})")

    ax.plot(xs, a_ih, "o-", color=BLUE,   lw=2.5, ms=6, label="Vanilla - Prune Ih_min to Ih_max")
    ax.plot(xs, b_ih, "s-", color=ORANGE, lw=2.5, ms=6, label="Ih tuned - Prune Ih_min to Ih_max")
    ax.fill_between(xs, a_ih, b_ih, color="gray", alpha=0.12, label="Gap")

    fs = cfg.get("font_size", 16)
    ax.set_xlabel(x_label, fontsize=fs, fontweight="bold")
    ax.set_title(
        f"I_h-Ordered Head Dropping: Vanilla vs I_h-Driven\n"
        f"({cfg['dataset']} dataset, lowest importance first)",
        fontsize=fs, fontweight="bold", pad=16,
    )
    if cfg.get("show_legend", True):
        ax.legend(fontsize=fs, loc="lower left")
    _apply_axis_cfg(ax, cfg, metric)

    if cfg["plot_ylim"] is None:
        all_vals = a_ih + b_ih
        ax.set_ylim(min(all_vals) - 0.01, max(ra["baseline"], rb["baseline"]) + 0.015)

    plt.tight_layout()
    path = os.path.join(cfg["out_dir"], f"{cfg['dataset']}_ih_order.png")
    fig.savefig(path, dpi=cfg["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure saved  →  {path}")


def plot_training_curves(history_a, history_b, cfg):
                                                                   
    metric = cfg["eval_metric"].upper()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg["fig_size"])

    ax1.plot(history_a["epoch"], history_a["train_loss"], "o-", color=BLUE,   lw=2, ms=4, label="Vanilla")
    ax1.plot(history_b["epoch"], history_b["train_loss"], "s-", color=ORANGE, lw=2, ms=4, label="I_h-Driven")
    ax1.axvline(x=cfg["warmup_epochs"], color="red", ls="--", alpha=0.5, label=f"Warmup end (ep {cfg['warmup_epochs']})")
    fs = cfg.get("font_size", 16)
    ax1.set_xlabel("Epoch", fontsize=fs); ax1.set_ylabel("Training Loss", fontsize=fs)
    ax1.set_title("Training Loss", fontsize=fs, fontweight="bold")
    ax1.tick_params(labelsize=fs)
    if cfg.get("show_legend", True):
        ax1.legend(fontsize=fs)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_a["epoch"], history_a["val_metric"], "o-", color=BLUE,   lw=2, ms=4, label="Vanilla")
    ax2.plot(history_b["epoch"], history_b["val_metric"], "s-", color=ORANGE, lw=2, ms=4, label="I_h-Driven")
    ax2.axvline(x=cfg["warmup_epochs"], color="red", ls="--", alpha=0.5)
    ax2.set_xlabel("Epoch", fontsize=fs); ax2.set_ylabel(f"Validation {metric}", fontsize=fs)
    ax2.set_title(f"Validation {metric}", fontsize=fs, fontweight="bold")
    ax2.tick_params(labelsize=fs)
    if cfg.get("show_legend", True):
        ax2.legend(fontsize=fs)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves — {cfg['dataset']}", fontsize=fs, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(cfg["out_dir"], f"{cfg['dataset']}_training_curves.png")
    fig.savefig(path, dpi=cfg["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure saved  →  {path}")


def plot_zoomed_comparison(results, cfg):
    

    metric = cfg["eval_metric"].upper()
    ra, rb = results["a"], results["b"]

    total = cfg["total_heads"]
    max_h = max(1, round(total * 0.99))               
    a_ih  = ra["auc_ih"][:max_h]
    b_ih  = rb["auc_ih"][:max_h]
    xs, x_label = _x_axis(max_h, cfg)

                                                           
    all_vals  = a_ih + b_ih
    y_max     = max(ra["baseline"], rb["baseline"]) + 0.003
    full_range = y_max - min(all_vals)
    y_min_zoom = y_max - full_range * 0.30                      

    fig, ax = plt.subplots(figsize=cfg["fig_size"])

    ax.axhline(y=ra["baseline"], color=BLUE,   ls="--", lw=2, alpha=0.65,
               label=f"Vanilla baseline ({ra['baseline']:.4f})")
    ax.axhline(y=rb["baseline"], color=ORANGE, ls="--", lw=2, alpha=0.65,
               label=f"Ih tuned baseline ({rb['baseline']:.4f})")

    ax.plot(xs, a_ih, "o-", color=BLUE,   lw=2.5, ms=6, label="Vanilla - Prune Ih_min to Ih_max")
    ax.plot(xs, b_ih, "s-", color=ORANGE, lw=2.5, ms=6, label="Ih tuned - Prune Ih_min to Ih_max")

    fs = cfg.get("font_size", 16)
    ax.set_xlabel(x_label, fontsize=fs, fontweight="bold")
    ax.set_title(
        f"Zoomed: I_h-Order Dropping (top 30% of range)\n"
        f"({cfg['dataset']} dataset)",
        fontsize=fs, fontweight="bold", pad=16,
    )
    if cfg.get("show_legend", True):
        ax.legend(fontsize=fs, loc="lower left")
    ax.set_ylim(y_min_zoom, y_max)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)

    plt.tight_layout()
    path = os.path.join(cfg["out_dir"], f"{cfg['dataset']}_zoomed.png")
    fig.savefig(path, dpi=cfg["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure saved  →  {path}")


def generate_all_plots(results, history_a, history_b, cfg):
    print("\n" + "─"*50)
    print("  Generating figures ...")
    print("─"*50)
    plot_all_strategies(results, cfg)
    plot_ih_order_only(results, cfg)
    plot_zoomed_comparison(results, cfg)
    if history_a["epoch"]:
        plot_training_curves(history_a, history_b, cfg)


def print_summary(results, cfg, test_a, test_b, best_val_a, best_val_b):
    ra, rb = results["a"], results["b"]
    metric = cfg["eval_metric"].upper()

    def heads_droppable(auc_list, baseline, threshold_pct=0.99):
        thr = baseline * threshold_pct
        last_ok = 0
        for i, v in enumerate(auc_list):
            if v >= thr:
                last_ok = i
        return last_ok

    hd_a = heads_droppable(ra["auc_ih"], ra["baseline"])
    hd_b = heads_droppable(rb["auc_ih"], rb["baseline"])

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY — {cfg['dataset'].upper()}")
    print(f"{'='*60}")
    print(f"  {'':30s}  {'Vanilla':>12}  {'I_h-Driven':>12}")
    print(f"  {'─'*58}")
    print(f"  {'Baseline ' + metric:30s}  {ra['baseline']:>12.4f}  {rb['baseline']:>12.4f}")
    print(f"  {'Best val ' + metric:30s}  {best_val_a:>12.4f}  {best_val_b:>12.4f}")
    print(f"  {'Test ' + metric:30s}  {test_a:>12.4f}  {test_b:>12.4f}")
    print(f"  {'Heads droppable @1% loss':30s}  {hd_a:>12d}  {hd_b:>12d}")
    print(f"  {'Final ' + metric + ' (all dropped)':30s}  {ra['auc_ih'][-1]:>12.4f}  {rb['auc_ih'][-1]:>12.4f}")
    print(f"  {'─'*58}")
    diff = hd_b - hd_a
    print(f"  I_h-Driven improvement: {diff:+d} heads at 1% loss threshold")
    print(f"{'='*60}\n")


def load_from_csv(csv_vanilla, csv_ih, cfg):
    

    va = pd.read_csv(csv_vanilla).sort_values("num_heads_dropped").reset_index(drop=True)
    ih = pd.read_csv(csv_ih).sort_values("num_heads_dropped").reset_index(drop=True)

                                                                                        
    baseline_a = float(va["baseline_auc"].iloc[0])
    baseline_b = float(ih["baseline_auc"].iloc[0])

                                                               
    auc_a = [baseline_a] + va["auc_after_drop"].tolist()
    auc_b = [baseline_b] + ih["auc_after_drop"].tolist()

                                                                   
    def extract_heads(df):
        if "layer" in df.columns and "head_index" in df.columns:
            imp_col = "importance_score" if "importance_score" in df.columns else None
            rows = []
            for _, r in df.iterrows():
                entry = {"layer": int(r["layer"]), "head": int(r["head_index"]),
                                     : float(r[imp_col]) if imp_col else 0.0}
                rows.append(entry)
            return rows
        return []

    sorted_heads_a = extract_heads(va)
    sorted_heads_b = extract_heads(ih)

                                                                
    total_heads = len(auc_a) - 1                                            
    cfg["total_heads"]  = total_heads
    cfg["num_heads"]    = cfg.get("num_heads", 8)
    cfg["num_layers"]   = cfg.get("num_layers", total_heads // cfg["num_heads"])

    cfg["eval_metric"] = "auc"               

                                                     
    if "dataset_name" in va.columns:
        cfg["dataset"] = str(va["dataset_name"].iloc[0])

    print(f"  ✓ Vanilla CSV : {csv_vanilla}  ({len(va)} rows, baseline={baseline_a:.4f})")
    print(f"  ✓ I_h CSV     : {csv_ih}  ({len(ih)} rows, baseline={baseline_b:.4f})")
    print(f"  ✓ total_heads={total_heads}  eval_metric={cfg['eval_metric']}")

                                                 
    results = {
           : {
                      :     baseline_a,
                    :       auc_a,
                        :   auc_a,                                  
                       :    auc_a,                                  
                          : sorted_heads_a,
                        :   {},
        },
           : {
                      :     baseline_b,
                    :       auc_b,
                        :   auc_b,
                       :    auc_b,
                          : sorted_heads_b,
                        :   {},
        },
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransTab Robustness Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",       default=None,  help="Dataset name (overrides CFG)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load saved checkpoints instead of training")
    parser.add_argument("--plot-only",     action="store_true",
                        help="Load saved results JSON and regenerate plots only")

                                                               
    parser.add_argument("--from-csv",      action="store_true",
                        help="Build plots directly from two CSV files — no model or JSON needed")
    parser.add_argument("--csv-vanilla",   default=None, metavar="PATH",
                        help="Path to vanilla head-dropping CSV")
    parser.add_argument("--csv-ih",        default=None, metavar="PATH",
                        help="Path to I_h-driven head-dropping CSV")

                                                                
    parser.add_argument("--ylim",  nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                        help="Y-axis limits, e.g. --ylim 0.78 0.83")
    parser.add_argument("--xlim",  nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                        help="X-axis limits (in %%), e.g. --xlim 0 80")
    parser.add_argument("--no-pct", action="store_true",
                        help="Use raw head count on X-axis instead of percentage")
    parser.add_argument("--font-size", type=int, default=None, metavar="N",
                        help="Font size for all plot text (default: 16)")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide legends on all figures")
    return parser.parse_args()


def main():
    args = parse_args()

                                
    if args.dataset:
        CFG["dataset"] = args.dataset
    if args.ylim:
        CFG["plot_ylim"] = args.ylim
    if args.xlim:
        CFG["plot_xlim"] = args.xlim
    if args.no_pct:
        CFG["plot_x_as_pct"] = False
    if args.font_size:
        CFG["font_size"] = args.font_size
    if args.no_legend:
        CFG["show_legend"] = False

    cfg = setup(CFG)

                                                              
    if args.from_csv:
        csv_va = args.csv_vanilla
        csv_ih = args.csv_ih
                                                                
        if not csv_va or not csv_ih:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ds = cfg["dataset"]
            if not csv_va:
                csv_va = os.path.join(script_dir, f"{ds}_headdroping_vanilla.csv")
            if not csv_ih:
                csv_ih = os.path.join(script_dir, f"{ds}_headdroping_ih_driven.csv")
        if not os.path.exists(csv_va):
            raise FileNotFoundError(f"Vanilla CSV not found: {csv_va}\n"
                                    f"Pass it explicitly with --csv-vanilla <path>")
        if not os.path.exists(csv_ih):
            raise FileNotFoundError(f"I_h CSV not found: {csv_ih}\n"
                                    f"Pass it explicitly with --csv-ih <path>")
        results = load_from_csv(csv_va, csv_ih, cfg)
        empty_history = {"epoch": [], "train_loss": [], "val_metric": []}
        generate_all_plots(results, empty_history, empty_history, cfg)
        print_summary(results, cfg, t_a=0.0, t_b=0.0,
                      best_val_a=results["a"]["baseline"],
                      best_val_b=results["b"]["baseline"])
        return

                                                               
    if args.plot_only:
        results, history_a, history_b, bv_a, bv_b, t_a, t_b = load_results(cfg)
        generate_all_plots(results, history_a, history_b, cfg)
        print_summary(results, cfg, t_a, t_b, bv_a, bv_b)
        return

                                                               
    import transtab
    trainset, valset, testset = load_dataset(cfg)
    cfg["_testset"] = testset                               

                                                               
    history_a = history_b = {"epoch": [], "train_loss": [], "val_metric": []}
    best_val_a = best_val_b = 0.0
    test_metric_a = test_metric_b = 0.0

    if args.skip_training:
        model_a, model_b = load_checkpoints(cfg)
    else:
        model_a, history_a, best_val_a, test_metric_a = train_vanilla(cfg, trainset, valset)
        model_b, history_b, best_val_b, test_metric_b = train_ih_driven(cfg, trainset, valset)

                                                               
    print("\n" + "─"*50)
    print("  EVALUATION PHASE")
    print("─"*50)

    results = {
           : evaluate_model(model_a, "Model A (Vanilla)",    testset, cfg),
           : evaluate_model(model_b, "Model B (I_h-Driven)", testset, cfg),
    }

                     
    del model_a, model_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

                                                               
    print("\n" + "─"*50)
    print("  SAVING RESULTS")
    print("─"*50)
    save_results(cfg, results, history_a, history_b,
                 best_val_a, best_val_b, test_metric_a, test_metric_b)

                                                               
    generate_all_plots(results, history_a, history_b, cfg)

                                                               
    print_summary(results, cfg, test_metric_a, test_metric_b, best_val_a, best_val_b)


if __name__ == "__main__":
    main()