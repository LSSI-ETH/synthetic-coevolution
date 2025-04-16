#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import EsmModel
from models.attn_modules import get_esm_model_str
import os

#----
# ESM-2 Backbone
#https://github.com/facebookresearch/esm

class Backbone_ESM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        model_str, self.esm_emb_dim = get_esm_model_str(args)

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

# classifier head
class PlmOnly(nn.Module):
    def __init__(self, 
                 input_size = 201,
                 input_emb_dim = 480, 
                 n_classes = 39, 
                 args = None,
                 device = None):
        super().__init__()
    
        self.device = device
        self.backbone = Backbone_ESM(args).to(self.device)

        input_dim = int( input_size * self.backbone.esm_emb_dim)
        
        self.linear = nn.Linear(input_dim, n_classes)
        self.flatten = nn.Flatten()

        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

    def forward(self, x, input_mask = None):
        x = self.backbone(x)
        x = self.flatten(x)
        attn = None
        return self.linear(x), attn