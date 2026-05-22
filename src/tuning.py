
                                                  
import torch


def freeze_gates(model):
                                                                      
    for layer in model.encoder.transformer_encoder:
        if hasattr(layer.self_attn, "head_gate_logits"):
            layer.self_attn.head_gate_logits.requires_grad_(False)


def set_gates_constant(model, num_layers, num_heads, value=1.0):
                                                 
    device = next(model.parameters()).device
    gates = torch.full((num_heads,), value, device=device)
    for li in range(num_layers):
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(gates.clone())


def update_gates_from_importance(model, importance_dict):
    

    normed = {}
    for li, scores in importance_dict.items():
        l2 = torch.norm(scores, p=2) + 1e-8
        normed[li] = scores / l2
        clamped = torch.clamp(normed[li], 1e-6, 1 - 1e-6)
        model.encoder.transformer_encoder[li].self_attn.set_head_masks(clamped)
    return normed
