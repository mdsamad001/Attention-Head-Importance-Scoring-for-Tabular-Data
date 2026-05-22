
                                              
import numpy as np
import torch
from tqdm import tqdm


def compute_importance(model, dataset, batch_size=32):
    

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
    n_samples = 0

    for start in tqdm(range(0, len(X), batch_size), desc="  Computing I_h", leave=False):
        end = min(start + batch_size, len(X))
        bx, by = X.iloc[start:end], y.iloc[start:end]

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


def importance_to_dict(importance_dict, num_layers):
                                                                                  
    raw = {}
    norm = {}
    for li in range(num_layers):
        scores = importance_dict[li]
        raw[str(li)] = scores.detach().cpu().tolist()
        l2 = torch.norm(scores, p=2) + 1e-8
        norm[str(li)] = (scores / l2).detach().cpu().tolist()
    return raw, norm
