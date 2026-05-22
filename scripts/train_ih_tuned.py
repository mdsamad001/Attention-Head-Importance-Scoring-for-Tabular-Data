
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import (
    load_dataset, build_model, evaluate_model,
    freeze_gates, set_gates_constant,
    compute_importance, importance_to_dict,
    update_gates_from_importance,
)

CFG = {
             :       "pc3",
                :    128,
             :       256,
                :    6,
               :     8,
             :       0.1,
                  :  50,
                   : 10,
                :    64,
        :            1e-4,
                  :  1e-5,
              :      10,
          :          42,
                   : 32,
}


def train(cfg, trainset, valset, testset):
    model = build_model(cfg)
    freeze_gates(model)
    set_gates_constant(model, cfg["num_layers"], cfg["num_heads"], value=1.0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    X_train, y_train = trainset
    X_val, y_val = valset
    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_metric, best_state, patience_ctr = -float("inf"), None, 0

    for epoch in range(cfg["total_epochs"]):
        model.train()
        indices = np.random.permutation(len(X_train))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(X_train), cfg["batch_size"]):
            idx = indices[i:i + cfg["batch_size"]]
            optimizer.zero_grad()
            _, loss = model(X_train.iloc[idx], y_train.iloc[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        model.eval()

                                            
        if epoch >= cfg["warmup_epochs"]:
            importance = compute_importance(model, trainset, cfg["ih_batch_size"])
            update_gates_from_importance(model, importance)

        val_auc = evaluate_model(model, X_val, y_val)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_metric:
            best_metric, patience_ctr = val_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        phase = "WARMUP" if epoch < cfg["warmup_epochs"] else "TUNING"
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg["warmup_epochs"]:
            print(f"  Epoch {epoch:3d} [{phase:7s}]  loss={avg_loss:.4f}  "
                  f"val_auc={val_auc:.4f}  patience={patience_ctr}/{cfg['patience']}")

        if patience_ctr >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(cfg["device"])
    freeze_gates(model)

          
    torch.save(best_state, cfg["ckpt_path"])
    print(f"\n  Checkpoint → {cfg['ckpt_path']}")

    X_test, y_test = testset
    test_auc = evaluate_model(model, X_test, y_test)
    print(f"  Best val_auc={best_metric:.4f}  test_auc={test_auc:.4f}")

                
    final_imp = compute_importance(model, testset, cfg["ih_batch_size"])
    raw, norm = importance_to_dict(final_imp, cfg["num_layers"])
    ih_data = {"dataset": cfg["dataset"], "model": "ih_tuned",
                       : {str(li): {"raw": raw[str(li)], "norm": norm[str(li)]}
                          for li in range(cfg["num_layers"])}}
    with open(cfg["ih_path"], "w") as f:
        json.dump(ih_data, f, indent=2)
    print(f"  Importance → {cfg['ih_path']}")

    history["best_val_auc"], history["test_auc"] = best_metric, test_auc
    with open(cfg["history_path"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History    → {cfg['history_path']}")

    return model


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",  default=None)
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--warmup",   type=int, default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    args = p.parse_args()

    if args.dataset:  CFG["dataset"] = args.dataset
    if args.epochs:   CFG["total_epochs"] = args.epochs
    if args.warmup:   CFG["warmup_epochs"] = args.warmup
    if args.lr:       CFG["lr"] = args.lr
    if args.patience: CFG["patience"] = args.patience

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir    = os.path.join(script_dir, "..", "checkpoints")
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ds = CFG["dataset"]
    CFG["device"]       = "cuda:0" if torch.cuda.is_available() else "cpu"
    CFG["ckpt_path"]    = os.path.join(ckpt_dir, f"{ds}_ih_tuned.pt")
    CFG["history_path"] = os.path.join(results_dir, f"{ds}_ih_tuned_history.json")
    CFG["ih_path"]      = os.path.join(results_dir, f"{ds}_ih_tuned_importance.json")

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    trainset, valset, testset, col_info = load_dataset(ds, CFG["seed"])
    CFG.update(col_info)

    print(f"\n{'='*50}")
    print(f"  Training I_h-Tuned — {ds}")
    print(f"  Warmup: {CFG['warmup_epochs']} epochs")
    print(f"{'='*50}\n")
    train(CFG, trainset, valset, testset)


if __name__ == "__main__":
    main()
