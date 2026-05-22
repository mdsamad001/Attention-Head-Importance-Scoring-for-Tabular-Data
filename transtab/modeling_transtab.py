
import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transtab import constants

import logging
logger = logging.getLogger(__name__)

class TransTabWordEmbedding(nn.Module):
    
       
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)
        return embeddings

class TransTabNumEmbedding(nn.Module):
    
       
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))           
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        

        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class TransTabFeatureExtractor:
    

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        

        if os.path.exists('./transtab/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./transtab/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel:                             
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

                                        
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, shuffle=False) -> Dict:
        

        encoded_inputs = {
                   :None,
                               :None,
                             :None,
                             :None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
                                                      
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

               
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0)                     
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']                     

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' '+ x) * x_mask                        
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols]                                                            
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0:                
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
                   
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

                        
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

                                   
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
                         : self.categorical_columns,
                    : self.binary_columns,
                       : self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
                   
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
                   
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

class TransTabFeatureProcessor(nn.Module):
    
       
    def __init__(self,
        vocab_size=None,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        device='cuda:0',
        ) -> None:
        

        super().__init__()
        self.word_embedding = TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
            )
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1,keepdim=True).to(embs.device)
            return embs

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None,
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs,
        ) -> Tensor:
        

        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        if x_num is not None and num_col_input_ids is not None:
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device))                                                  
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            num_feat_embedding = self.align_layer(num_feat_embedding)

        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0:                      
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0],dtype=int)[:,None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

                               
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class MaskedMultiHeadAttention(nn.Module):
    

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first
        self.scale = math.sqrt(self.head_dim)
        
                           
        self.q_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
                                                                                   
        self.head_gate_logits = nn.Parameter(torch.zeros(nhead, **factory_kwargs))
        
                                                                           
        self._head_outputs = None
        self._store_head_outputs = False
        
    def get_head_masks(self):
                                                        
        return torch.sigmoid(self.head_gate_logits)
    
    def set_head_masks(self, mask_values):
                                                                    
                                                 
        mask_values = torch.clamp(mask_values, 1e-6, 1 - 1e-6)
        logits = torch.log(mask_values / (1 - mask_values))                   
        self.head_gate_logits.data = logits
        
    def enable_head_output_storage(self, enable=True):
                                                                             
        self._store_head_outputs = enable
        if not enable:
            self._head_outputs = None
            
    def get_stored_head_outputs(self):
                                                             
        return self._head_outputs
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        

        if self.batch_first:
            bs, seq_len, _ = query.shape
        else:
            seq_len, bs, _ = query.shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
                                                              
        Q = self.q_proj(query).view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
                                                         
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
                                            
        if key_padding_mask is not None:
                                                            
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
        
                                          
        if attn_mask is not None:
            scores = scores + attn_mask
        
                             
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
                                                      
        head_outputs = torch.matmul(attn_weights, V)
        
                                                     
        if self._store_head_outputs:
                                                    
            head_outputs.retain_grad()
            self._head_outputs = head_outputs
        
                                                  
        xi = torch.sigmoid(self.head_gate_logits)            
        masked_head_outputs = head_outputs * xi.view(1, -1, 1, 1)             
        
                                                                         
        combined = masked_head_outputs.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        
                                 
        output = self.out_proj(combined)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, None                                                  
    
    def compute_head_importance(self):
        

        if self._head_outputs is None or self._head_outputs.grad is None:
            return None
        
                                                  
        head_out = self._head_outputs
        grad = self._head_outputs.grad
        
                                                                                     
        importance = torch.sum(head_out * grad, dim=(0, 2, 3)).abs()
        
        return importance.detach()
    
    def get_sparsity_loss(self):
                                                                   
        xi = torch.sigmoid(self.head_gate_logits)
        return xi.sum()


class TransTabTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True, use_head_mask=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.use_head_mask = use_head_mask
        self.nhead = nhead
        
        if use_head_mask:
                                                        
            self.self_attn = MaskedMultiHeadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=batch_first, **factory_kwargs)
        else:
                                            
            self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first,
                                                **factory_kwargs)
                                             
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

                                 
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

                                                        
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

                          
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        if self.use_head_mask:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               )[0]
        else:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               )[0]
        return self.dropout1(x)
    
    def get_head_masks(self):
                                                             
        if self.use_head_mask and hasattr(self.self_attn, 'get_head_masks'):
            return self.self_attn.get_head_masks()
        return None
    
    def set_head_masks(self, mask_values):
                                                             
        if self.use_head_mask and hasattr(self.self_attn, 'set_head_masks'):
            self.self_attn.set_head_masks(mask_values)
    
    def enable_head_output_storage(self, enable=True):
                                                                     
        if self.use_head_mask and hasattr(self.self_attn, 'enable_head_output_storage'):
            self.self_attn.enable_head_output_storage(enable)
    
    def compute_head_importance(self):
                                            
        if self.use_head_mask and hasattr(self.self_attn, 'compute_head_importance'):
            return self.self_attn.compute_head_importance()
        return None
    
    def get_sparsity_loss(self):
                                               
        if self.use_head_mask and hasattr(self.self_attn, 'get_sparsity_loss'):
            return self.self_attn.get_sparsity_loss()
        return torch.tensor(0.0)

                        
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g           
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask= None, src_key_padding_mask= None, is_causal=None, **kwargs) -> Tensor:
        

        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:                        
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x

class TransTabInputEncoder(nn.Module):
    

    def __init__(self,
        feature_extractor,
        feature_processor,
        device='cuda:0',
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        

        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
                                
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

                              
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

class TransTabEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        use_head_mask=True,
        ):
        super().__init__()
        self.num_layer = num_layer
        self.num_attention_head = num_attention_head
        self.use_head_mask = use_head_mask
        
                                                           
        self.transformer_encoder = nn.ModuleList([
            TransTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
                use_head_mask=use_head_mask,
            )
            for _ in range(num_layer)
        ])

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        
           
        outputs = embedding
        for layer in self.transformer_encoder:
            outputs = layer(outputs, src_key_padding_mask=attention_mask)
        return outputs
    
    def get_all_head_masks(self):
        

        masks = {}
        for i, layer in enumerate(self.transformer_encoder):
            mask = layer.get_head_masks()
            if mask is not None:
                masks[i] = mask
        return masks
    
    def set_all_head_masks(self, masks_dict):
        

        for i, layer in enumerate(self.transformer_encoder):
            if i in masks_dict:
                layer.set_head_masks(masks_dict[i])
    
    def enable_head_output_storage(self, enable=True):
                                                        
        for layer in self.transformer_encoder:
            layer.enable_head_output_storage(enable)
    
    def compute_all_head_importance(self):
        

        importance = {}
        for i, layer in enumerate(self.transformer_encoder):
            imp = layer.compute_head_importance()
            if imp is not None:
                importance[i] = imp
        return importance
    
    def get_total_sparsity_loss(self):
                                                        
        total_loss = torch.tensor(0.0)
        for layer in self.transformer_encoder:
            loss = layer.get_sparsity_loss()
            if loss is not None:
                total_loss = total_loss + loss
        return total_loss
    
    def print_head_masks_summary(self):
                                                  
        masks = self.get_all_head_masks()
        print("\n" + "="*60)
        print("HEAD MASK SUMMARY (ξ_h values)")
        print("="*60)
        for layer_idx, mask in masks.items():
            mask_values = mask.detach().cpu().numpy()
            print(f"\nLayer {layer_idx}:")
            for h, val in enumerate(mask_values):
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"  Head {h}: [{bar}] {val:.4f}")
        print("="*60 + "\n")

class TransTabLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:]                               
        x = self.norm(x)
        logits = self.fc(x)
        return logits
    
class TransTabLinearRegressor(nn.Module):
    def __init__(self,
        hidden_dim=128) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:]                               
        x = self.norm(x)
        output = self.fc(x)
        return output

class TransTabProjectionHead(nn.Module):
    def __init__(self,
        hidden_dim=128,
        projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h

class TransTabCLSToken(nn.Module):
           
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs

class TransTabModel(nn.Module):
    

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None:
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )
        
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
            )

        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            )

        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        

        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

                                                          
        encoder_output = self.encoder(**embeded)

                       
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding

    def load(self, ckpt_dir):
        

        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

                                
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        

        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

                                           
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        

        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

                                                                       
    def get_head_masks(self):
        

        return self.encoder.get_all_head_masks()
    
    def set_head_masks(self, masks_dict):
        

        self.encoder.set_all_head_masks(masks_dict)
    
    def enable_head_importance_tracking(self, enable=True):
        

        self.encoder.enable_head_output_storage(enable)
    
    def compute_head_importance(self):
        

        return self.encoder.compute_all_head_importance()
    
    def get_sparsity_loss(self):
        

        return self.encoder.get_total_sparsity_loss()
    
    def print_head_summary(self):
                                                         
        self.encoder.print_head_masks_summary()
    
    def prune_heads(self, threshold=0.1):
        

        pruned = {}
        masks = self.get_head_masks()
        for layer_idx, mask in masks.items():
            pruned_heads = (mask < threshold).nonzero(as_tuple=True)[0].tolist()
            if pruned_heads:
                new_mask = mask.clone()
                new_mask[mask < threshold] = 0.0
                self.encoder.transformer_encoder[layer_idx].set_head_masks(new_mask)
                pruned[layer_idx] = pruned_heads
        return pruned


class TransTabClassifier(TransTabModel):
    

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None):
        

        if isinstance(x, dict):
                                                       
            inputs = x
        elif isinstance(x, pd.DataFrame):
                                
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'TransTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

                                                              
        encoder_output = self.encoder(**outputs)                           

                    
        logits = self.clf(encoder_output)

        if y is not None:
                                         
            if self.num_class == 2:
                y_ts = torch.tensor(y.values).to(self.device).float()
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss
    
class TransTabRegressor(TransTabModel):
    

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=1,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.regressor = TransTabLinearRegressor(hidden_dim=hidden_dim)
        
        self.loss_fn = nn.MSELoss()
        self.to(device)

    def forward(self, x, y=None):
        

        if isinstance(x, dict):
                                                       
            inputs = x
        elif isinstance(x, pd.DataFrame):
                                
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'TransTabRegressor takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

                                                              
        encoder_output = self.encoder(**outputs)                           

                    
        output = self.regressor(encoder_output)

        if y is not None:
                                     
            y_ts = torch.tensor(y.values).to(self.device).float()
            loss = self.loss_fn(output.flatten(), y_ts)
            loss = loss.mean()
        else:
            loss = None

        return output, loss

class TransTabForCL(TransTabModel):
    

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=True,
        temperature=10,
        base_temperature=10,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
            )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        

        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_positive_pairs(x, self.num_partition)
            for sub_x in sub_x_list:
                                                   
                feat_x = self.input_encoder(sub_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                feat_x_proj = feat_x[:,0,:]                     
                feat_x_proj = self.projection_head(feat_x_proj)                     
                feat_x_list.append(feat_x_proj)
        elif isinstance(x, dict):
                                 
            for input_x in x['input_sub_x']:
                feat_x = self.input_encoder.feature_processor(**input_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                feat_x_proj = feat_x[:, 0, :]
                feat_x_proj = self.projection_head(feat_x_proj)
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame or dict(pretokenized), get {type(x)} instead')

        feat_x_multiview = torch.stack(feat_x_list, axis=1)                      

        if y is not None and self.supervised:
                                  
            y = torch.tensor(y.values, device=feat_x_multiview.device)
            loss = self.supervised_contrastive_loss(feat_x_multiview, y)
        else:
                                                       
            loss = self.self_supervised_contrastive_loss(feat_x_multiview)
        return None, loss

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def self_supervised_contrastive_loss(self, features):
        

        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
                                    
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
                          
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
                                                      
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def supervised_contrastive_loss(self, features, labels):
        

        labels = labels.contiguous().view(-1,1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)

                                
        anchor_feature = contrast_feature
        anchor_count = contrast_count

                        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
                                 
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
                   
        mask = mask.repeat(anchor_count, contrast_count)
                                      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
                          
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
                                                      
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
