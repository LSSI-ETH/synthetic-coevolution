#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchmetrics.classification import MatthewsCorrCoef, MultilabelF1Score
import logging

#===========================   Dataset & Loaders        ================================ 
class TorchDataset(Dataset):
    """
    Converts categorically encoded sequences & labels into a torch Dataset
    
    Parameters
    ----------
    encoded_seqs: list
        categorically encoded protein or nucleotide sequences
    labels: list
        class labels or regression fitness values corresponding to sequences

    Returns
    -------
    tuple of sequences, labels (y)
    """    
    def __init__(self, args, encoded_seqs, labels, transform=None):
        
        if 'esm' in args.rbd_plm_backbone:
            self.encoded_seqs = encoded_seqs['input_ids']
        else:
            self.encoded_seqs = encoded_seqs
        self.labels = labels
        self.transform = transform
        self.args = args
        
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        
        seq = self.encoded_seqs[idx]
        
        if 'esm' in self.args.dataset:
            #seq = np.array(seq)
            seq = torch.tensor(np.array(seq))
            
        label = self.labels[idx]
        mask = None
        masked_indices = None
        
        if self.transform:
            mask, masked_indices = self.transform(label)
            
        return seq, label, mask, masked_indices
    
#===========================   Collater Fn to Apply Padding         ====================

class Collater(object):
    """
    
    Parameters
    ----------
    alphabet: str
        vocabulary size (i.e. amino acids, nucleotide ngrams). used for one-hot encoding dimension calculation
    pad_tok: float 
        padding token. zero padding is used as default
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine one-hot or categorical encoding

    Returns
    -------
    padded sequences, labels (y)
    """    
    def __init__(self, vocab_length: int, 
                pad_tok=0,
                args = None):        
        self.vocab_length = vocab_length
        self.pad_tok = pad_tok
        self.args = args

    def __call__(self, batch):

        sequences, y, mask, masked_indices= zip(*batch)

        if mask[0] is not None: mask = torch.stack(mask)
        if masked_indices[0] is not None: masked_indices = torch.stack(masked_indices)
        
        y = np.array(y)
        y = torch.tensor(y).squeeze()
        y = y.type(torch.FloatTensor)
        
        if 'esm' not in self.args.dataset:
            maxlen = sequences[0].shape[0]
            padded = torch.stack([torch.cat([i, i.new_zeros(maxlen - i.size(0))], 0) for i in sequences],0)

            if 'transformer' not in self.args.basemodel and 'rbd_plm' not in self. args.basemodel:
                padded = F.one_hot(padded, num_classes = self.vocab_length)
                    
        elif 'esm' in self.args.dataset:
            maxlen = sequences[0].shape[0]
            padded = torch.stack([torch.cat([i, i.new_zeros(maxlen - i.size(0))], 0) for i in sequences],0)
    
            if 'transformer' not in self.args.basemodel and 'rbd_plm' not in self.args.basemodel:
                padded = torch.unsqueeze(padded, dim =1)

        return padded, y, mask, masked_indices



#===========================   Convert Data to torch.DataLoader        ======================

def data_to_loader(x, y, is_test, args):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    x: list
        categorically encoded protein or nucleotide training, validation, and testing sequences
    y: pandas.core.series.Series
        class labels corresponding to training, validation, & testing sequences
    batch_size: int
        batch size to be used for dataloader
    args: argparse.ArgumentParser
        arguments specified by user. 
    Returns
    -------
    torch DataLoader objects for training, validation, testing, meta sets
    """    
    
    batch_size = args.batch_size
    
    y = y.to_list()
    
    #load hugging face tokenizer for appropriate esm model
    if 'esm' in args.rbd_plm_backbone:

        model_str, _ = get_esm_model_str(args) #returns esm model_str, emb_dim
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        x = tokenizer(x, return_tensors = 'pt')
   
    if is_test == False and 'rbd_plm' in args.basemodel:
        data = TorchDataset(args, x, y, 
                            transform = flip_known_labels_to_unk(args))
        
    elif 'rbd_plm'  in args.basemodel and is_test == True:
        data = TorchDataset(args, x, y, transform = None)
        
    else:
        data = TorchDataset(args, x, y, transform = None)
    
    if len(y) % args.batch_size != 0:
        
        drop_last_bool = True
    else:
        drop_last_bool = False
    
    if is_test == True: shuffle_bool = False
    elif is_test == False: shuffle_bool = True
        
    vocab_length = 21
    
    num_works = 0

    if args.non_block == True:    
        pin_mem = True
    else:
        pin_mem = False
        
    if args.distributed == True and is_test == False:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        shuffle_bool = False
    else:
        sampler = None
    
    data_loader = torch.utils.data.DataLoader(data, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle_bool, 
                                              num_workers = num_works, 
                                              pin_memory = pin_mem,
                                              collate_fn=Collater(vocab_length = vocab_length, 
                                                                  pad_tok=0., 
                                                                  args=args
                                                                  ), 
                                              drop_last=drop_last_bool,
                                              sampler = sampler)
    
    return data_loader, sampler 


#===========================   Dataset Transforms        ================================ 

class flip_known_labels_to_unk(object):
    '''
        masks a fraction (args.mask_fraction) of known labels for modified rbd_plm model
    '''
    def __init__(self, args):
        self.args = args
    #known_label_mask = labels != -1
    #labels = torch.tensor(labels)
    #known_labels = (labels != -1).view(labels.size(0) * labels.size(1))
    
    
    def __call__(self, labels):
        
        labels = torch.tensor(labels)
        known_labels = (labels != -1)#.view(labels.size(0))
        known_labels = known_labels.nonzero(as_tuple=True)[0]
        known_idxs = torch.randperm(len(known_labels))
        
        #mask lmt fraction of known labels
        
        randvar = torch.rand(1)
        
        if len(known_idxs) >= self.args.lmt_threshold and randvar >= self.args.initiate_lmt_threshold:
            mask_fraction = int(np.floor(self.args.lmt_mask_fraction * len(known_idxs)))
        else:
            mask_fraction = len(known_idxs)
            
        unk_mask_indices = known_labels[known_idxs[:mask_fraction]] 
        
        masked_labels = labels.clone()
        masked_labels.scatter_(0,unk_mask_indices,-1)
    
        masked_indices = torch.ones(torch.numel(torch.arange(len(labels))), 
                                    dtype=torch.bool)
        
        #tensor with False at locations of masked labels & True elsewhere
        masked_indices[unk_mask_indices] = False 
        
        return masked_labels, masked_indices




#===========================   Categorically Encode ngrams    ==========================

def encode_ngrams(x,args):
    """
    Converts amino acid or nucleotide sequences to categorically encoded vectors based on a chosen
    encoding approach (ngram vocabulary).    
    
    Parameters
    ----------
    x: pandas.core.series.Series
        pandas Series containing strings of protein or nucleotide training, validation, or testing sequences
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine correct vocabulary size, output 
        shape, and if a mask should be returned

    Returns
    -------
    x_train_idx, x_val_idx, x_test_idx: list
        categorically encoded sequences
    vocabulary:
        vocabulary used for ngram encoding. to be passed to dataloaer & collate functions
    """    
    def seq_to_cat(seq_df, word_to_idx_dictionary):
        '''
        input: dataframe of sequences & dictionary containing tokens in vocabulary
        output: out_idx: list of torch.Tensors of categorically encoded (vocab index) ngrams 
        '''
        out_idxs = []
        
        if isinstance(seq_df,pd.Series): seq_df = seq_df.to_list()
            
        for i in range(len(seq_df)): out_idxs.append(torch.tensor([word_to_idx_dictionary[w] for w in seq_df[i] if w != None and w != '' ], dtype=torch.long))
        
        return out_idxs

    vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 
                  'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']    
    
    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    x_idx = seq_to_cat(x, word_to_ix)
    
    return x_idx



#============== ESM Backbone Model Configuration ===============
def get_esm_model_str(args):
    #return esm string & embedding dimension to load tokenizer & backbone
    if 'esm' in args.rbd_plm_backbone:
        esm_dict = {}
        esm_dict['protbert'] = ['Rostlab/prot_bert', 1024]
        esm_dict['esm_8m'] = ['facebook/esm2_t6_8M_UR50D', 320]
        esm_dict['esm_35m'] = ['facebook/esm2_t12_35M_UR50D', 480]
        esm_dict['esm_150m'] = ['facebook/esm2_t30_150M_UR50D', 640]
        esm_dict['esm_650m'] = ['facebook/esm2_t33_650M_UR50D', 1280]
        esm_dict['esm_3b'] = ['facebook/esm2_t36_3B_UR50D', 2560]
        
        try:
            return esm_dict[args.rbd_plm_backbone][0], esm_dict[args.rbd_plm_backbone][1]
        except:
            raise ValueError(f'esm model string {args.rbd_plm_backbone} not'\
                             ' recognized. Options: esm_8m, esm_35m, esm_150m'\
                                 'esm_650m, esm_3b') 
    else:
        return args.rbd_plm_backbone, None
    



#==============          VOCs         ===================
#Taft, Weber et al: https://doi.org/10.1016/j.cell.2022.08.024
#He et al: https://doi.org/10.1016/j.xcrm.2023.100991 

def get_voc_data_loader(voc_str, args):

    if voc_str == 'he':
        num_voc_seqs = 12
        data_str = 'he_paper_voc'
        
    elif voc_str == 'taft':
        num_voc_seqs = 36
        data_str = 'taft_weber_vocs'
    else:
        raise Exception('Uknown VOC ID string',voc_str) 
    
    if 'esm' in args.dataset:
        data = pd.read_pickle(f'{args.train_path}{data_str}_650m_esm.pkl')
    else:
        data = pd.read_csv(f'{args.train_path}/{data_str}.csv')
        
    
    if len(data) < args.batch_size:
        #use math.gcd instead of math.lmc for python < 3.9
        def lcm(a,b):
            return (a * b) // math.gcd(a,b)
        
        mult_factor = lcm(args.batch_size, num_voc_seqs)
        data = pd.concat([data.copy()]*mult_factor, 
                         ignore_index = True) #if test set is smaller than batch size
        
    if 'esm' in args.dataset:
        x_test = data['embedding']
    else:
        x_test = data['aa_seq']
    
    if 'esm' not in args.dataset and 'esm' not in args.rbd_plm_backbone:
        x_test = encode_ngrams(x_test, args)
    elif 'esm' in args.dataset or 'esm' in args.rbd_plm_backbone:
        x_test = x_test.to_list()

        
    label_cols = ['2C08wt', '2C08a', '2C08b', '2C08d', '2C08O', '87wt', '87a', '87b',
    '87d', '555wt', '555a', '555d', '33wt', '33a', '33b', '33d', '33O',
    '50wt', '50a', '50b', '50d', '16wt', '16a', '16d', '64wt', '64a', '64b',
    '64d', '82wt', '82a', '82b', '82O', 'A23581', 'GR32A4', 'LY1404',
    'S2E12', 'S2H97', 'S309', 'ACE2',]
    
    #taft_weber_labels = ['555wt','16wt','87wt','33wt']
    #label_col_idxs =[9, 21, 5, 12]
    
    
    #get true labels from test set data dataframe
    if voc_str == 'he':
        label_cols = list(data.columns[3:9])
    elif voc_str == 'taft':
        label_cols = list(data.columns[1:5])
    label_df= pd.DataFrame(data = data, columns = [label_cols[i] for i in range(len(label_cols))])
    
    label_df['label_vector'] = label_df.values.tolist()
    y_test = label_df['label_vector']

    test_loader, _ = data_to_loader(x = x_test, y=y_test, 
                                    is_test = True, args = args)
    
    return test_loader


#============== Compute Metrics on Taft, Weber et al VOCs ===============
def voc_metrics(voc_data_loader, model,  device, args, voc_str, is_test = False):
    '''
    compute metrics on VOC data
    model should be BaseModel trainer wrapper class
    '''
    
    
    if voc_str == 'he':
        number_labels = len(model.idx_he_voc)
        num_voc_seqs = 12
        voc_idxs = model.idx_he_voc
    elif voc_str == 'taft':
        number_labels = len(model.idx_taft_voc)
        num_voc_seqs = 36
        voc_idxs = model.idx_taft_voc
    else:
        raise Exception('Uknown VOC ID string',voc_str) 
        
        
    model.model.eval()
    
    
    mcc = MatthewsCorrCoef(task = 'multilabel', num_labels = number_labels, 
                           ignore_index = -1, validate_args=False).to(device)
    
    f1_macro = MultilabelF1Score(num_labels = number_labels, multidim_average = 'global',
                           average='macro', ignore_index = -1, validate_args=False).to(device)
    
    f1_micro = MultilabelF1Score(num_labels = number_labels, multidim_average = 'global',
                           average='micro', ignore_index = -1, validate_args=False).to(device)
    
    f1_weighted = MultilabelF1Score(num_labels = number_labels, multidim_average = 'global',
                           average='weighted', ignore_index = -1, validate_args=False).to(device)
    
    f1_none = MultilabelF1Score(num_labels = number_labels, multidim_average = 'global',
                           average='none', ignore_index = -1, validate_args=False).to(device)
    
    with torch.no_grad():
        
        for batch_idx, (inputs, labels, mask, _) in enumerate(voc_data_loader):
            inputs = inputs.to(device, non_blocking = args.non_block)
            labels.to(device, non_blocking = args.non_block)
            
            if 'rbd_plm' in args.basemodel:
                mask = -1 * torch.ones( 
                    ( args.batch_size), 39).to(device, non_blocking = args.non_block)
                #mask = model.replace_label_values(labels, -1,-1,-1).to(device, non_blocking = args.non_block)
                pred, attns = model.get_predictions(model.model, inputs, mask = mask)                
    
            else:
                pred, _ = model.get_predictions(model.model, inputs)              
            
            if batch_idx == 0:
                labels_voc = labels.long().to(model.device, non_blocking = args.non_block)
                pred_voc = pred[:,voc_idxs]
                
                pred_voc = pred_voc[:num_voc_seqs,:] 
                labels_voc = labels_voc[:num_voc_seqs,:]
                
                batch_mcc = mcc(pred_voc,labels_voc)
                batch_f1_mtr = f1_macro(pred_voc,labels_voc) 
                batch_f1_micro = f1_micro(pred_voc,labels_voc) 
                batch_f1_weighted = f1_weighted(pred_voc,labels_voc) 
                batch_f1_none = f1_none(pred_voc,labels_voc) 
                
        metrics = {}
        
        #================ Full Data Metrics per Epoch =========================
        metrics[f'{voc_str}_mcc'] = mcc.compute()
        
        metrics[f'{voc_str}_f1_macro'] = f1_macro.compute()
        metrics[f'{voc_str}_f1_micro'] = f1_micro.compute()
        metrics[f'{voc_str}_f1_weighted'] = f1_weighted.compute()
        metrics[f'{voc_str}_f1_none'] = f1_none.compute()
        
        
        epoch_mcc = metrics[f'{voc_str}_mcc']
    
    if is_test:
        return metrics