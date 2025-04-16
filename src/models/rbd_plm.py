#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from transformers import EsmModel
import numpy as np
from models.attn_modules import self_attn_block, inter_attn_layer, get_esm_model_str
import os

#----
# RBD-pLM

class RBD_pLM(nn.Module):
    def __init__(self,args,
                 device,
                 num_labels,
                 input_size,
                 use_lmt,
                 pos_emb=True,
                 heads=4,
                 emb_dim = 480,
                 ):
        super().__init__()
        
        self.use_lmt = use_lmt
        self.device = device
        self.input_size = input_size
        self.num_labels = num_labels
        self.args = args
        self.dropout = args.dropout
        self.emb_dim = emb_dim
        self.heads = heads
        self.use_pos_enc = pos_emb
        self.hidden = self.emb_dim * self.args.hdim_expansion
        assert args.rbd_plm_backbone in ['emb','esm_8m', 
                                         'esm_35m', 'esm_150m', 
                                         'esm_650m',]
        #===========================================================
        # Backbone
        if 'esm' in args.rbd_plm_backbone:
            self.backbone = Backbone_ESM(args).to(self.device)
            self.emb_dim = self.backbone.esm_emb_dim
            self.sequence_length = 201
            self.hidden = self.emb_dim * self.args.hdim_expansion
        
        if self.args.rbd_plm_backbone == 'emb':
            self.backbone = torch.nn.Embedding(input_size, 
                                               self.emb_dim, 
                                               padding_idx = None)
            self.sequence_length = input_size
            
        
        if 'esm' not in self.args.rbd_plm_backbone: 
            self.backbone.apply(self.init_weights)
        
        #======================================================================
        # Label & Position Embeddings
        
        self.unknown_token = 81
        if args.use_lmt == True:

            self.label_emb = torch.nn.Embedding(int(num_labels * 2 + 4), self.emb_dim, 
                                           padding_idx= self.unknown_token)
            self.label_emb.apply(self.init_weights)
            
        # Position Embeddings
        if self.use_pos_enc:

            self.position_embeddings_features = nn.Embedding(self.sequence_length + self.num_labels, 
                                                           self.emb_dim)
            self.position_embeddings_features.apply(self.init_weights)    
            self.dropout_pe_feats = nn.Dropout(self.dropout)
                
        # Transformers
        layer_inter_attn = inter_attn_layer

        self.inter_attn = nn.ModuleList(
            [layer_inter_attn(d_model = self.emb_dim, 
                                n_heads=heads,
                                hdim = self.hidden,
                                activation = self.args.activation,
                                batch_first = True,
                                dropout = self.dropout,
                                args = self.args
                                ) for _ in range(self.args.inter_attn_layers)])
    
        assert args.combined_attn_layers > 0
        
        self.combined_self_attn = nn.ModuleList(
            [self_attn_block(d_model = self.emb_dim,
                            n_heads = heads,
                            hdim = self.hidden,
                            activation=self.args.activation,
                            batch_first=True,
                            dropout=self.dropout,
                            args = self.args
                            ) for _ in range(self.args.combined_attn_layers)])
        self.combined_self_attn.apply(self.init_weights)
        
        self.LayerNormIntermediate = nn.LayerNorm(self.emb_dim)
        self.LayerNormIntermediate.apply(self.init_weights)
        
        self.inter_attn.apply(self.init_weights)
        
        # Classifier               
        self.output_linear = ElementwisePoolingLayer(input_dim = self.num_labels,
                                                output_dim = self.emb_dim,
                                                bias = True,
                                                )

    def init_weights(self,module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            stdv = 1. / math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
            #module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if isinstance(module, nn.LayerNorm) and module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self,src,mask):
        
        BS = src.size(0)

        if self.backbone is not None:
            features = self.backbone(src)
        else:
            features = src.unsqueeze(1)

        label_self_attn_mask = None
        features_src_cross_attn_mask = None
        combined_attn_mask = None
        if self.use_lmt:
            
            # Create mask for unknown labels
            unknown_label_mask = mask == self.unknown_token

            # create self attention mask of shape (batch_size * attn_heads, num_labels) to prevent attending to unknown tokens
            label_self_attn_mask = unknown_label_mask
            label_self_attn_mask = unknown_label_mask.unsqueeze(1).repeat(1, mask.shape[1], 1)
            label_self_attn_mask = label_self_attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1).view(BS * self.heads, mask.shape[1], mask.shape[1])

            # create cross attention mask of shape (batch_size * attn_heads, num_labels, seq_len) to treat unknown tokens as padding
            features_src_cross_attn_mask = unknown_label_mask.unsqueeze(1).repeat(1, features.shape[1], 1)
            features_src_cross_attn_mask = features_src_cross_attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1).view(BS * self.heads, features.shape[1], mask.shape[1])

            # create combined attention mask
            truncated_sequence = src[:,1:-1] # remove BOE & EOS tokens
            combined_seq_label_tensor = torch.cat((truncated_sequence, mask), dim=1)
            combined_attn_mask = (combined_seq_label_tensor == self.unknown_token)
            combined_attn_mask[:,:truncated_sequence.shape[1]] = False
            combined_attn_mask = combined_attn_mask.unsqueeze(1).repeat(1, truncated_sequence.shape[1] + mask.shape[1], 1)
            combined_attn_mask = combined_attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1).view(BS * self.heads, truncated_sequence.shape[1] + mask.shape[1], truncated_sequence.shape[1] + mask.shape[1])
            
            label_embeddings = self.label_emb(mask) 
            
        if self.use_pos_enc:
            
            # Positional Encoding on both Seq & Label features
            labels_and_sequence = torch.cat([features,label_embeddings],dim = 1)
            position_ids_features = torch.arange(0, labels_and_sequence.size(1), dtype=torch.long, device=self.device)
            position_ids_features = position_ids_features.unsqueeze(0).repeat(BS, 1) 
            position_embeddings_features = self.position_embeddings_features(position_ids_features)

            labels_and_sequence += position_embeddings_features
            labels_and_sequence = self.dropout_pe_feats(labels_and_sequence)
            features = labels_and_sequence[:,0:self.sequence_length,:]    
            label_embeddings = labels_and_sequence[:,-label_embeddings.size(1):,:]    

        for inter_attn in self.inter_attn:
            label_embeddings, features = inter_attn(Lf = label_embeddings,
                                                    Sf = features,
                                                    self_attn_label_mask = label_self_attn_mask,
                                                    cross_attn_label_mask = features_src_cross_attn_mask,
                                                    )
        attns = None
        embeddings = torch.cat([features,label_embeddings],dim = 1)
        embeddings = self.LayerNormIntermediate(embeddings)

        attns = []
        for combined_self_attn in self.combined_self_attn:
            embeddings, attn = combined_self_attn(embeddings, attn_mask = combined_attn_mask)
            if attn is not None:
                attns += attn.detach().unsqueeze(0).data
            else:
                attns = None

        label_embeddings = embeddings[:,-label_embeddings.size(1):,:]    
        features = embeddings[:,0:self.sequence_length,:]    
        embeddings = label_embeddings
        
        return self.output_linear(embeddings), attns


# Elementwise weighted pooling layer https://arxiv.org/abs/1908.07325

class ElementwisePoolingLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 bias=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.input_dim))
        else:
            self.register_parameter('bias', None)
        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.input_dim):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.input_dim):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x * self.weight
        x = torch.sum(x,2)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    

class Backbone_ESM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        model_str, self.esm_emb_dim = get_esm_model_str(args)
        self.args = args
        
        if os.environ['TRANSFORMERS_OFFLINE']== '1':
           path = os.environ['HOME']
           model_str = f"{path}/.cache/huggingface/models/{model_str}"
        
        self.base_network = EsmModel.from_pretrained(model_str,
                                                        output_attentions = args.return_esm_attns,
                                                        output_hidden_states = False,
                                                        )
        for param in self.base_network.parameters():
            param.requires_grad = False
        for param in self.base_network.pooler.parameters(): 
                param.requires_grad=False
        for param in self.base_network.contact_head.parameters(): 
                param.requires_grad=False
        
    def forward(self,x):
        x = self.base_network(x).last_hidden_state
        x = x[:,1:-1,:] #omit bos & eos tokens
        return x

    