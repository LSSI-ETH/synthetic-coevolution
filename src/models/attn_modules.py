#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP_Res_Block(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hid_dim, 
                 activation = 'relu',
                 dropout=0.1,
                 args = None):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)        
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, in_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation=='gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def _ffwd_block(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(self.activation(x))

    def forward(self, x):
        x = x + self._ffwd_block(self.layer_norm(x))
        return x
    
class self_attn_block(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads,
                 hdim = None,
                 activation = 'relu',
                 batch_first = True,
                 dropout=0.1,
                 args = None):
        super().__init__()
        
        if hdim == None:
            hdim = d_model

        self.return_attns = False
        if args is not None:
            self.return_attns = args.return_attns
            
        self.self_attn = nn.MultiheadAttention(d_model, 
                                         n_heads, 
                                         dropout=dropout,
                                         batch_first=batch_first,
                                         )

        self.MLP_Res_Block = MLP_Res_Block(d_model,
                                                hdim,
                                                activation,
                                                dropout,
                                                args)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask = None):
        
        norm1_x = self.norm1(x)

        if not self.return_attns:
            x2 = self.self_attn(norm1_x, norm1_x, norm1_x, need_weights = False,
                                attn_mask = attn_mask)[0]
            attn = None
        else:
            x2, attn = self.self_attn(norm1_x, norm1_x, norm1_x, need_weights = True,
                                attn_mask = attn_mask)
        x2 = self.dropout(x2)
        x = x + x2
        x = self.MLP_Res_Block(x)
        
        return x, attn

class cross_attn_block(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads,
                 hdim = None,
                 activation = 'relu',
                 batch_first = True,
                 dropout=0.1, 
                 args = None):
        super().__init__()
        
        if hdim == None:
            hdim = d_model
            
        self.mha = nn.MultiheadAttention(d_model, 
                                         n_heads, 
                                         dropout=dropout,
                                         batch_first=batch_first,
                                         )
        if activation=='gelu':
            self.activation = nn.GELU()
            
        else:
            self.activation = nn.ReLU()

        self.MLP_Res_Block = MLP_Res_Block(d_model,
                                                hdim,
                                                activation,
                                                dropout,
                                                args)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, attn_mask = None):
        
        mem = self.norm1(mem)
        x2 = self.mha(self.norm2(x), mem, mem, need_weights = False,
                      attn_mask = attn_mask)[0]
        x2 = self.dropout(x2)
        x = x + x2
        x = self.MLP_Res_Block(x)
        return x
        
    
class inter_attn_layer(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads=4, 
                 hdim = None,
                 activation = 'relu',
                 batch_first = True,
                 dropout=0.1,
                 args = None):
        super().__init__()
        
        if hdim == None:
            hdim = d_model
        
        self.emb_dim = d_model

        self.L_self_attn_layer = self_attn_block(d_model,
                                                 n_heads,
                                                 hdim,
                                                 activation=activation,
                                                 batch_first=True,
                                                 dropout=dropout,
                                                 args = args)
        
        self.S_self_attn_layer = self_attn_block(d_model,
                                                 n_heads,
                                                 hdim,
                                                 activation=activation,
                                                 batch_first=True,
                                                 dropout=dropout,
                                                 args = args)
        
        
        self.L_cross_attn = cross_attn_block(d_model,
                                                n_heads,
                                                hdim,
                                                activation=activation,
                                                batch_first=True,
                                                dropout=dropout,
                                                args = args)
        
        self.S_cross_attn = cross_attn_block(d_model,
                                                 n_heads,
                                                 hdim,
                                                 activation=activation,
                                                 batch_first=True,
                                                 dropout=dropout,
                                                 args = args)
        
        self.norm1 = nn.LayerNorm(d_model)
        
    def forward(self, 
                Lf, #label features
                Sf, #sequence features
                self_attn_label_mask = None,
                cross_attn_label_mask = None):

        BS, labels_length, emb_dim = Lf.shape
        assert emb_dim == self.emb_dim
        BS, sequence_length, emb_dim = Sf.shape
        assert emb_dim == self.emb_dim

        merged_embeddings = torch.cat([Sf, Lf], dim = 1)
        merged_embeddings = self.norm1(merged_embeddings)
        
        Sf = merged_embeddings[:,0:sequence_length,:]    
        Lf = merged_embeddings[:,-labels_length:,:]    
        
        Lf, _ = self.L_self_attn_layer(Lf, attn_mask = self_attn_label_mask)
        Sf, _ = self.S_self_attn_layer(Sf, attn_mask = None)
        
        Lf = self.L_cross_attn(Lf, Sf, attn_mask = None)
        Sf = self.S_cross_attn(Sf, Lf, attn_mask = cross_attn_label_mask)

        return Lf, Sf

#============== ESM Backbone Model Configuration ===============
def get_esm_model_str(args):
    
    ''' load desired esm model & attributed from transformers library '''
    
    if 'esm' in args.rbd_plm_backbone:
        esm_dict = {}
        esm_dict['esm_8m'] = ['facebook/esm2_t6_8M_UR50D', 320]
        esm_dict['esm_35m'] = ['facebook/esm2_t12_35M_UR50D', 480]
        esm_dict['esm_150m'] = ['facebook/esm2_t30_150M_UR50D', 640]
        esm_dict['esm_650m'] = ['facebook/esm2_t33_650M_UR50D', 1280]
        esm_dict['esm_3b'] = ['facebook/esm2_t36_3B_UR50D', 2560]
        
        try:
            return esm_dict[args.rbd_plm_backbone][0], esm_dict[args.rbd_plm_backbone][1]
        except:
            raise ValueError(f'esm model string {args.rbd_plm_backbone} not recognized') 
    else:
        return args.rbd_plm_backbone, None