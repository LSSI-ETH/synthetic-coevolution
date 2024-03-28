#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNN(nn.Module):
    
    def __init__(self, input_size, 
                 conv_filters, 
                 dense_nodes,
                 n_out, 
                 kernel_size, 
                 dropout, 
                 args):
        
        super().__init__()
        
        pad, dilation, stride_var = 0, 1, 2
        maxpool_kernel = 2
        mp_stride = maxpool_kernel
        mp_pad = 0
        input_vector_len = 21
            
        conv_out_size  = math.floor( ( ( input_vector_len + 2*pad - dilation*(kernel_size - 1) - 1 ) /stride_var) + 1)            
        mp_out_size = math.floor( ( ( conv_out_size + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        transition_nodes =  math.floor( (conv_filters ) * mp_out_size)
        
        conv_layers = [nn.Conv1d(input_size, conv_filters, kernel_size = kernel_size, padding=pad, stride = stride_var, bias=False)]
        conv_layers.append(nn.BatchNorm1d(conv_filters))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=maxpool_kernel, padding = mp_pad))
        self.conv_bn_relu_stack = nn.Sequential(*conv_layers)
                           
        self.flatten = nn.Flatten()        
        self.dropout = nn.Dropout(p=dropout)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(transition_nodes, dense_nodes), 
            nn.ReLU(),
            )
        self.out_layer = nn.Linear(dense_nodes,n_out)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
        torch.nn.init.xavier_uniform_(self.out_layer.weight)
        self.out_layer.bias.data.zero_()

    def forward(self, x, mask = None):
        x = x.float()
        if mask is not None:
            x = x * mask
        x = self.conv_bn_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.dropout(x)

        return self.out_layer(x)


class LogisticRegression(nn.Module):
    def __init__(self, 
                 input_size, 
                 n_classes, 
                 args):
        super().__init__()
    
        input_vector_len = 21 # amino acid vocabulaary for one-hot encoding
        input_dim = int( input_size * input_vector_len)
        self.linear = nn.Linear(input_dim, n_classes)
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, input_mask = None):
        x = self.flatten(x)
        return self.linear(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, args, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.args = args
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :]            
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, 
                 ntoken, 
                 emb_dim, 
                 nhead, 
                 nhid, 
                 nlayers, 
                 n_classes, 
                 seq_len, 
                 args, 
                 dropout=0.1,):
        super().__init__()
        
        self.args = args
        self.pos_encoder = PositionalEncoding(emb_dim, args, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, 
                                                 dropout, batch_first=True,
                                                 norm_first = True, 
                                                 activation = 'relu')
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, emb_dim)
        self.flatten = nn.Flatten()
        self.decoder = nn.Linear(emb_dim * seq_len, 1024)
        self.emb_dim = emb_dim
        self.relu = nn.ReLU()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)        
        self.out_layer = nn.Linear(1024, n_classes)
        
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights) 
        
        torch.nn.init.kaiming_normal_(self.decoder.weight,
                                      nonlinearity = 'relu')
        self.decoder.bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.out_layer.weight)
        self.out_layer.bias.data.zero_()

    def init_weights(self,module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            stdv = 1. / math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, src, input_mask = None):
        src = self.encoder(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src)
        if input_mask is not None:
            output = self.transformer_encoder(src, src_key_padding_mask = input_mask)
        elif input_mask == None:
            output = self.transformer_encoder(src)
        output = self.flatten(output)
        output = self.decoder(output)
        output = self.relu(output)
        output = self.dropout(output)
        return self.out_layer(output)
    