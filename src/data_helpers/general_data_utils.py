#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import Dataset
import torch.nn.functional as F

import os

# uncomment if using cached transformer model weights
#path = os.environ['HOME']
#os.environ['TORCH_HOME'] = path
#os.environ['HF_HOME']=f"{path}/.cache/huggingface"
#os.environ['TRANSFORMERS_CACHE']=f"{path}/.cache/huggingface/models"
#os.environ['TRANSFORMERS_OFFLINE']= '1'
os.environ['TRANSFORMERS_OFFLINE']= '0'


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
            seq = torch.tensor(np.array(seq))
        label = np.asarray(self.labels[idx])
        
        return seq, label
    
#===========================   Collater Fn to Apply Padding         ====================

class Collater(object):
    """
    
    Encodes input strings (amino acids) to encoded vectors, either one-hot or categorical.
    For RBD-pLM, Collater also applies masked label modeling (MLM) to the labels.

    Parameters
    ----------
    alphabet: str
        vocabulary size (i.e. amino acids). used for one-hot encoding dimension calculation
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
                args = None,
                tokenizer = None):        
        self.vocab_length = vocab_length
        self.pad_tok = pad_tok
        self.args = args
        self.tokenizer = tokenizer
        self.token_dict_length = 80
        self.unknown_token = 81
        self.mask_token = 80
        self.token_dict = self.create_token_dictionary(self.token_dict_length) # 2 states per label (0,1) plus -1 for unknown
        
    def create_token_dictionary(self, vocabulary_size):
        '''
        Creates a token dictionary for the labels. Each of the 39 labels has 2 possible states: 0, 1. 
            -1 is a general, unknown token.
            Ex: label 0 negative -> 00, label 0 positive -> 01, label 3 negative -> 30, label 3 positive -> 31, 
        '''
        tokens = [int(f"{i // 2}{i % 2}") for i in range(vocabulary_size)]
        token_dict = {token: i for i, token in enumerate(tokens)}
        token_dict[-1] = self.unknown_token
        token_dict[-100] = self.mask_token # inference: switch all tokens to mask token
        return token_dict
    
    def convert_labels_to_tokens(self, labels):
        
        '''
        Convert labels to tokens.
        Args:
            labels (list): A list of labels.
        Returns:
            list: A list of tokens as torch.Tensors.
        '''

        tokens = []
        for row in labels:
            row_tokens = []
            for i, val in enumerate(row):
                if val == -1:
                    row_tokens.append(self.token_dict[-1]) 
                else:
                    row_tokens.append(self.token_dict[(i+1) * 10 + int(val)])
            tokens.append(row_tokens)
        return torch.Tensor(tokens).long()
        
    def __call__(self, batch):
        
        sequences, labels = zip(*batch)

        labels = np.array(labels)
        labels = torch.tensor(labels).squeeze()
        labels = labels.type(torch.FloatTensor)
                
        maxlen = sequences[0].shape[0]
        padded = torch.stack([torch.cat([i, i.new_zeros(maxlen - i.size(0))], 0) for i in sequences],0)
        
        if 'transformer' not in self.args.basemodel and 'rbd_plm' not in self. args.basemodel:
            padded = F.one_hot(padded, num_classes = self.vocab_length)
                    
        # masked label modeling
        masked_labels, masked_label_indices = None, None
        if self.args.basemodel == 'rbd_plm':
            label_toks = self.convert_labels_to_tokens(labels.clone())
            known_labels = (label_toks != self.unknown_token) # not equal to unknown token
            masked_labels = label_toks.clone()
            # mask fraction of known labels for lmt based on threshold

            randvar = torch.rand(1)
            if randvar >= self.args.initiate_lmt_threshold:
        
                prob_mat = torch.full(known_labels.shape, self.args.lmt_mask_fraction)
                prob_mat.masked_fill_(~known_labels, value=0.0)
                masked_label_indices = torch.bernoulli(prob_mat).bool()

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(torch.full(masked_labels.shape, 0.8)).bool() & masked_label_indices
                masked_labels[indices_replaced] = self.mask_token # set to mask token

                # 10% of the time, we replace masked input tokens with random token
                indices_random = torch.bernoulli(torch.full(masked_labels.shape, 0.5)).bool() & masked_label_indices & ~indices_replaced
                random_tokens = torch.randint(self.mask_token, masked_label_indices.shape, dtype=torch.long)
                masked_labels[indices_random] = random_tokens[indices_random]

            else:
                masked_label_indices = known_labels
                masked_labels[masked_label_indices] = self.mask_token # set to unknown 

        return padded, labels, masked_labels, masked_label_indices


#===========================   Convert Data to torch.DataLoader        ======================

def x_y_to_dataset(x, y, is_test, args):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    x: list
        categorically encoded protein training, validation, and testing sequences
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
    
    y = y.to_list()
    
    #load hugging face tokenizer for appropriate esm model
    if 'esm' in args.rbd_plm_backbone:

        model_str, _ = get_esm_model_str(args) #returns esm model_str, emb_dim
        
        if os.environ['TRANSFORMERS_OFFLINE']== '1' and 'rbd_plm' in args.basemodel:
            path = os.environ['HOME']
            model_str = f"{path}/.cache/huggingface/models/{model_str}"
        
        tokenizer = AutoTokenizer.from_pretrained(model_str)

        x = tokenizer(x, return_tensors = 'pt')
   
    if is_test == False and 'rbd_plm' in args.basemodel:

        data = TorchDataset(args, x, y, transform = None) 
        
    elif 'rbd_plm'  in args.basemodel and is_test == True:
        data = TorchDataset(args, x, y, transform = None)
        
    else:
        data = TorchDataset(args, x, y, transform = None)
    return data

def dataset_to_dataloader(data, is_test, args, tokenizer = None):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    data: torch.utils.data.Dataset
        torch Dataset object containing protein sequences and their labels
    args: argparse.ArgumentParser
        arguments specified by user. 
    Returns
    -------
    torch DataLoader objects for training, validation, testing, meta sets
    """    
    
    batch_size = args.batch_size
    
    drop_last_bool = False
    if len(data) % args.batch_size != 0:
        drop_last_bool = True
    
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
    
    if 'rbd_plm_mlm' in args.basemodel:
        
        model_str, _ = get_esm_model_str(args) #returns esm model_str, emb_dim
        tokenizer = None
        
        if 'esm' in args.rbd_plm_backbone:
            if os.environ['TRANSFORMERS_OFFLINE']== '1':
                path = os.environ['HOME']
                model_str = f"{path}/.cache/huggingface/models/{model_str}"
            tokenizer = AutoTokenizer.from_pretrained(model_str)

    data_loader = torch.utils.data.DataLoader(data, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle_bool, 
                                              num_workers = num_works, 
                                              pin_memory = pin_mem,
                                              collate_fn=Collater(vocab_length = vocab_length, 
                                                                  pad_tok=0., 
                                                                  args=args,
                                                                  tokenizer = tokenizer), #tokenizer for mlm 
                                              drop_last=drop_last_bool,
                                              sampler = sampler)
    
    return data_loader, sampler 





#===========================   Categorically Encode ngrams    ==========================

def encode_ngrams(x,args):
    """
    Converts amino acids categorically encoded vectors based on a chosen
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

def convert_labels_to_tokens(labels, token_dict):
    
    '''
    Convert labels to tokens.
    Args:
        labels (list): A list of labels.
    Returns:
        list: A list of tokens as torch.Tensors.
    '''

    tokens = []
    for row in labels:
        row_tokens = []
        for i, val in enumerate(row):
            if val == -1:
                row_tokens.append(token_dict[-1]) 
            else:
                row_tokens.append(token_dict[(i+1) * 10 + val])
        tokens.append(row_tokens)
    return torch.Tensor(tokens).long()



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
# Taft, Weber et al: https://doi.org/10.1016/j.cell.2022.08.024
# He et al: https://doi.org/10.1016/j.xcrm.2023.100991 

def get_voc_data_loader(voc_str, args):

    if voc_str == 'he':
        num_voc_seqs = 12
        data_str = 'he_paper_vocs'
        
    elif voc_str == 'taft':
        num_voc_seqs = 36
        data_str = 'taft_weber_vocs'
    else:
        raise Exception('Uknown VOC ID string',voc_str) 
    

    data = pd.read_csv(f'{args.train_path}/{data_str}.csv')
        
    
    if len(data) < args.batch_size:
        #use math.gcd instead of math.lmc for python < 3.9
        def lcm(a,b):
            return (a * b) // math.gcd(a,b)
        
        mult_factor = lcm(args.batch_size, num_voc_seqs)
        data = pd.concat([data.copy()]*mult_factor, 
                         ignore_index = True) #if test set is smaller than batch size
        
    x_test = data['aa_seq']
    
    if 'esm' not in args.rbd_plm_backbone:
        x_test = encode_ngrams(x_test, args)
    else:
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

    test_data = x_y_to_dataset(x = x_test, y=y_test, 
                                    is_test = True, args = args)
    
    test_loader, _ = dataset_to_dataloader(test_data, is_test = True, args = args)

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
        
        for batch_idx, (inputs, labels, mask, _,) in enumerate(voc_data_loader):
            inputs = inputs.to(device, non_blocking = args.non_block)
            labels.to(device, non_blocking = args.non_block)
            
            if 'rbd_plm' in args.basemodel:
                mask = 81 * torch.ones(39).to(device, non_blocking = args.non_block) # set all labels to unknown
                mask[voc_idxs] = 80 # set voc targets to mask token
                mask = mask.long()
                mask = mask.unsqueeze(0).repeat(args.batch_size, 1)    
                pred, attns = model.get_predictions(model.model, inputs, masked_labels = mask)    
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