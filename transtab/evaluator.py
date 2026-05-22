
from collections import defaultdict
import os
import pdb

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from transtab import constants

def predict(clf, 
    x_test,
    y_test=None,
    return_loss=False,
    eval_batch_size=256,
    ):
    

    clf.eval()
    pred_list, loss_list = [], []
    for i in range(0, len(x_test), eval_batch_size):
        bs_x_test = x_test.iloc[i:i+eval_batch_size]
        bs_y_test = y_test.iloc[i:i+eval_batch_size] if y_test is not None else None
        with torch.no_grad():
            logits, loss = clf(bs_x_test, bs_y_test)
        
        if loss is not None:
            loss_list.append(loss.item())
            
        if logits.shape[-1] == 1:                        
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else:                             
            pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
            
    pred_all = np.concatenate(pred_list, 0)
        
    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    if return_loss:
        avg_loss = np.mean(loss_list)
        return avg_loss
    else:
        return pred_all

def evaluate(ypred, y_test, metric='auc', seed=123, bootstrap=False):
    np.random.seed(seed)
    eval_fn = get_eval_metric_fn(metric)
    res_list = []
    stats_dict = defaultdict(list)
    if bootstrap:
        for i in range(10):
            sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
            sub_ypred = ypred[sub_idx]
            sub_ytest = y_test.iloc[sub_idx]
            try:
                sub_res = eval_fn(sub_ytest, sub_ypred)
            except ValueError:
                print('evaluation went wrong!')
            stats_dict[metric].append(sub_res)
        for key in stats_dict.keys():
            stats = stats_dict[key]
            alpha = 0.95
            p = ((1-alpha)/2) * 100
            lower = max(0, np.percentile(stats, p))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print('{} {:.2f} mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
            if key == metric: res_list.append((upper+lower)/2)
    else:
        res = eval_fn(y_test, ypred)
        res_list.append(res)
    return res_list

def get_eval_metric_fn(eval_metric):
    fn_dict = {
             : acc_fn,
             : auc_fn,
             : mse_fn,
                  : None,
    }
    return fn_dict[eval_metric]

def acc_fn(y, p):
    y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)

def auc_fn(y, p):
    y = np.asarray(y)
    p = np.asarray(p)

                                                           
    if p.ndim == 1:
        return roc_auc_score(y, p)

                                                                          
    if p.ndim == 2 and p.shape[1] == 2:
        return roc_auc_score(y, p[:, 1])

                                
    if p.ndim == 2 and p.shape[1] > 2:
        return roc_auc_score(y, p, multi_class="ovr", average="macro")

    raise ValueError(f"Unsupported prediction shape for AUC: {p.shape}")

def mse_fn(y, p):
    return mean_squared_error(y, p)

class EarlyStopping:
                                                                                             
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0:                
            self.early_stop = False
            return
        
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
                                                        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, constants.WEIGHTS_NAME))
        self.val_loss_min = val_loss

