
                                           
import random
import numpy as np
from tqdm import tqdm

from .importance import compute_importance


def _evaluate(model, X, y):
                                        
    import transtab
    preds = transtab.predict(model, X)
    return transtab.evaluate(preds, y, metric="auc")[0]


def _save_masks(model, num_layers):
    saved = {}
    for li in range(num_layers):
        saved[li] = model.encoder.transformer_encoder[li].self_attn.get_head_masks().clone()
    return saved


def _restore_masks(model, saved):
    for li, mask in saved.items():
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(mask)


def _drop_in_order(model, testset, heads_order, num_layers):
                                                                        
    X, y = testset
    saved = _save_masks(model, num_layers)
    baseline = _evaluate(model, X, y)
    aucs = [baseline]

    for h in tqdm(heads_order, desc="  Dropping heads", leave=False):
        li, hi = h["layer"], h["head"]
        mask = model.encoder.transformer_encoder[li].self_attn.get_head_masks().clone()
        mask[hi] = 0.0
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(mask)
        aucs.append(_evaluate(model, X, y))

    _restore_masks(model, saved)
    return aucs


def evaluate_all_strategies(model, testset, cfg, seed=42):
    

    num_layers = cfg["num_layers"]
    ih_bs = cfg.get("ih_batch_size", 32)

    importance = compute_importance(model, testset, batch_size=ih_bs)

    all_heads = [
        {"layer": li, "head": hi, "importance": float(score)}
        for li, scores in importance.items()
        for hi, score in enumerate(scores.cpu().numpy())
    ]
    sorted_heads = sorted(all_heads, key=lambda h: h["importance"])

                                       
    print("    → I_h order (min to max) ...")
    auc_ih = _drop_in_order(model, testset, sorted_heads, num_layers)

                  
    random_heads = sorted_heads.copy()
    random.seed(seed)
    random.shuffle(random_heads)
    print("    → Random order ...")
    auc_random = _drop_in_order(model, testset, random_heads, num_layers)

                                        
    reverse_heads = sorted_heads[::-1]
    print("    → Worst first (max to min) ...")
    auc_worst = _drop_in_order(model, testset, reverse_heads, num_layers)

    return {
                  :     auc_ih[0],
                :       auc_ih,
                    :   auc_random,
                   :    auc_worst,
                      : sorted_heads,
                    : {
            str(li): scores.detach().cpu().tolist()
            for li, scores in importance.items()
        },
    }
