
                                                
import numpy as np
import torch


def load_dataset(dataset_name, seed=42):
    

    import transtab
    transtab.random_seed(seed)

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols =        transtab.load_data(dataset_name)

    X_train, y_train = trainset
    num_class = len(np.unique(y_train.values if hasattr(y_train, "values") else y_train))

    col_info = {
                  :  cat_cols,
                  :  num_cols,
                  :  bin_cols,
                   : num_class,
    }

    print(f"  Train={len(X_train)}  Val={len(valset[0])}  Test={len(testset[0])}")
    print(f"  Classes={num_class}\n")

    return trainset, valset, testset, col_info


def build_model(cfg):
                                                         
    import transtab
    transtab.random_seed(cfg.get("seed", 42))

    return transtab.build_classifier(
        categorical_columns=cfg["cat_cols"],
        numerical_columns=cfg["num_cols"],
        binary_columns=cfg["bin_cols"],
        num_class=cfg["num_class"],
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layer=cfg.get("num_layers", 6),
        num_attention_head=cfg.get("num_heads", 8),
        hidden_dropout_prob=cfg.get("dropout", 0.1),
        ffn_dim=cfg.get("ffn_dim", 256),
        device=cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
    )


def evaluate_model(model, X, y):
                                       
    import transtab
    preds = transtab.predict(model, X)
    return transtab.evaluate(preds, y, metric="auc")[0]
