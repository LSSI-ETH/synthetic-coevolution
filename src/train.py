#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler

from models.models import CNN, Transformer, LogisticRegression
from models.plm_only import PlmOnly
from models.rbd_plm import RBD_pLM
from loss_functions import ResampleLoss

from torchmetrics import MatthewsCorrCoef
from torchmetrics.classification import MultilabelF1Score, MultilabelJaccardIndex, MultilabelHammingDistance
from torchmetrics.functional.classification import binary_matthews_corrcoef

from transformers import AdamW
import logging
import wandb

class Trainer(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args
        self.num_classes = 39 # number labels
        self.input_size = 201 # rbd sequence length
        self.ntokens = 21 # number amino acids 
        
        #---
        # base model selection
        if args.basemodel == 'cnn':
            filters, dense  = 512, 1024
            self.model = CNN(input_size = self.input_size, 
                             conv_filters = filters, 
                             dense_nodes = dense, 
                             n_out = self.num_classes, 
                             kernel_size = 3, 
                             dropout = args.dropout, 
                             args = args).to(self.device)

        elif args.basemodel == 'transformer' :
            embedding_dim = 320
            hidden_dim = embedding_dim
            
            num_layers = 3
            if args.dataset == 'ed_3':
                num_layers = 2
            
            self.model = Transformer(ntoken = self.ntokens, 
                                     emb_dim = embedding_dim, nhead = 4, 
                                     nhid = hidden_dim, nlayers = num_layers, 
                                     n_classes = self.num_classes, 
                                     seq_len = self.input_size, 
                                     args = args, dropout = args.dropout
                                     ).to(self.device)
            
        elif args.basemodel == 'logistic_regression' :
            self.model = LogisticRegression(input_size = self.input_size, 
                                            n_classes = self.num_classes,
                                            args = args).to(self.device).to(self.device)
        

        elif args.basemodel == 'rbd_plm':
            embedding_dim = 128
            self.model = RBD_pLM(args = self.args,
                                 device = self.device,
                                 num_labels = self.num_classes,
                                 input_size = self.input_size,
                                 use_lmt = args.use_lmt,
                                 pos_emb=True,
                                 heads=4,
                                 emb_dim = embedding_dim).to(self.device)


        elif args.basemodel == 'rbd_plm_lr': # plm only
            self.model = PlmOnly(input_size = 201,
                               input_emb_dim = 320, 
                               n_classes = 39, 
                               args = args,
                               device = self.device)
    
        #---
        # optimizer
        self.learning_rate = args.learn_rate
        if args.opt_id == 'sgd': 
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=args.learn_rate, 
                                             momentum=0.9)
            
        elif args.opt_id == 'adam':  
            betas = (0.9, 0.999)
            eps = 1e-8
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # no weight decay for bias & layernorm 
            no_decay = ['bias', 'LayerNorm.weight']
            decay_params = [p for n, p in param_dict.items() if not any(nd in n for nd in no_decay)]
            nodecay_params = [p for n, p in param_dict.items() if any(nd in n for nd in no_decay)]
            optimizer_grouped_parameters = [
                {'params': decay_params, 'weight_decay': args.weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
                ]
            
            self.optimizer = AdamW(optimizer_grouped_parameters, 
                                          lr=args.learn_rate, 
                                          betas=betas, 
                                          eps = eps,
                                          correct_bias = not self.args.bert_adam, #https://arxiv.org/pdf/2006.05987.pdf
                                          no_deprecation_warning = True)
                     
        else: 
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=args.learn_rate, 
                                             momentum=0.9)
        #---
        # loss function
        if args.loss_fn == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction = 'none').to(self.device)
        
        elif args.loss_fn == 'focal_loss':
            train_num = self.args.train_len
            class_freq = self.args.class_freq
            self.loss_fn = ResampleLoss(reweight_func=None, loss_weight=1.0,
                         focal=dict(focal=True, alpha=0.5, gamma=2),
                         logit_reg=dict(),
                         class_freq=class_freq, train_num=train_num, device = self.device)
            
        elif args.loss_fn == 'dbloss':
            train_num = self.args.train_len
            class_freq = self.args.class_freq
            self.loss_fn = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.02, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num, device = self.device)
            
        # define label bucket indices according to training set label frequency
        # necessary for loss functions: focal_loss, dbloss
        self.idx_head = args.label_idx_high
        self.idx_mid = args.label_idx_mid
        self.idx_tail = args.label_idx_tail 
        self.idx_he_voc = [34, 37, 5, 9, 12, 38]
        self.idx_taft_voc = [9, 21, 5, 12]
        
        logging.info(f'args.dataset: {args.dataset}')
        logging.info(f'args.label_idx_high: {args.label_idx_high}')
        logging.info(f'args.label_idx_mid: {args.label_idx_mid}')
        logging.info(f'args.label_idx_tail: {args.label_idx_tail}')
        
        self.scaler = GradScaler(enabled = args.use_amp)
            
    # handle predictions for different model types
    def get_predictions(self,model, X, mask = None):
        attns = None
        if 'transformer' in self.args.basemodel:        
            pred = model(X)
        elif 'rbd_plm' in self.args.basemodel:
            pred, attns = model(X, mask)
        else:       
            pred = model(X.float())
        return pred, attns
    
    
    def replace_label_values(self,tensor,on_neg_1,on_zero,on_one):
        # for rbd_plm label masking during inference 
        # as well as ignoring loss for missing labels during backprop
        res = tensor.clone()
        res[tensor==-1] = on_neg_1
        res[tensor==0] = on_zero
        res[tensor==1] = on_one
        return res
    
    def train_step(self,train_dataloader, epoch, is_rank0_wandb, batch_size, train_sampler):
        
        if self.args.distributed == True:
            train_sampler.set_epoch(epoch)
            
        train_loss = 0   
        self.model.train()
        
        self.optimizer.zero_grad(set_to_none=True)

        for batch, (X, labels, mask, ignored_label_idxs) in enumerate(train_dataloader):

            X = X.to(self.device, non_blocking = self.args.non_block)
            labels = labels.to(self.device, non_blocking = self.args.non_block)
            
            with autocast(enabled = self.args.use_amp):
                if 'rbd_plm' in self.args.basemodel:
                    mask = mask.to(self.device, non_blocking = self.args.non_block)
                    ignored_label_idxs = ignored_label_idxs.to(self.device, non_blocking = self.args.non_block)

                    pred, _ = self.get_predictions(self.model, X, mask = mask)

                else:
                    pred, _ = self.get_predictions(self.model, X)                

                if self.args.loss_fn != 'bce':
                    loss = self.loss_fn(pred,labels, 
                                        rbd_plm_ignored_label_idxs = ignored_label_idxs,
                                        args = self.args)                

                if self.args.loss_fn == 'bce':
                    loss = self.loss_fn(pred,labels)
                    
                    if 'rbd_plm' in self.args.basemodel:
                        
                        if self.args.use_lmt == True:
                            unkown_label_mask = self.replace_label_values(labels, 0,1,1)
                            loss = (unkown_label_mask * loss)
                            
                            if self.args.loss_reduce == 'mean':
                                loss = loss.sum() / unkown_label_mask.sum()
                            elif self.args.loss_reduce == 'sum':
                                loss = loss.sum()
                            
                            
                        elif self.args.use_lmt == False:
                            unkown_label_mask = self.replace_label_values(labels, 0,1,1)
                            loss = (unkown_label_mask * loss)
                            
                            if self.args.loss_reduce == 'mean':
                                loss = loss.sum() / unkown_label_mask.sum()
                            elif self.args.loss_reduce == 'sum':
                                loss = loss.sum()
                            
                    else:
                        unkown_label_mask = self.replace_label_values(labels, 0,1,1)
                        loss = (unkown_label_mask * loss)
                        
                        if self.args.loss_reduce == 'mean':
                            loss = loss.sum() / unkown_label_mask.sum()
                        elif self.args.loss_reduce == 'sum':
                            loss = loss.sum()
                        
            loss = loss / self.args.grad_accumulation_steps
            train_loss += loss
            self.scaler.scale(loss).backward()
            
            #check for unused parameters
            #for name, param in self.model.named_parameters():
                #if param.grad is None and param.requires_grad == True:
                    #logging.info(name)
                    #print(name)
            
            if ((batch + 1) % self.args.grad_accumulation_steps == 0) or (batch + 1 == len(train_dataloader)):
                self.scaler.step(self.optimizer) 
                self.scaler.update()        
                self.optimizer.zero_grad(set_to_none=True) 
        
        if is_rank0_wandb and self.args.wandb_logging:
            wandb.log({"train/train_loss": train_loss})
            wandb.log({'epoch': epoch, 
                       'train_loss': train_loss})
        
        print(f'Train Loss: {train_loss}')
        logging.info(f'Train Loss: {train_loss}')
            
    def test_step(self, test_loader,epoch, is_rank0_wandb, is_test = False):
        
        self.model.eval()
        
        test_loss = 0
        val_args = False
        
        #=========================== Full Dataset Metrics =====================
        f1_macro = MultilabelF1Score(num_labels = self.num_classes, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_micro = MultilabelF1Score(num_labels = self.num_classes, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_weighted = MultilabelF1Score(num_labels = self.num_classes, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_none = MultilabelF1Score(num_labels = self.num_classes, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)

        hamming_macro = MultilabelHammingDistance(num_labels = self.num_classes, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        hamming_micro = MultilabelHammingDistance(num_labels = self.num_classes, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        hamming_weighted = MultilabelHammingDistance(num_labels = self.num_classes, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        hamming_none = MultilabelHammingDistance(num_labels = self.num_classes, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)
        
        jaccard_macro = MultilabelJaccardIndex(num_labels = self.num_classes, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        jaccard_micro = MultilabelJaccardIndex(num_labels = self.num_classes, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        jaccard_weighted = MultilabelJaccardIndex(num_labels = self.num_classes, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        jaccard_none = MultilabelJaccardIndex(num_labels = self.num_classes, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)
        
        mcc = MatthewsCorrCoef(task = 'multilabel', num_labels = self.num_classes, 
                               ignore_index = -1, validate_args=val_args).to(self.device)
        
        
        
        #=========================== Head Label Metrics =====================
        head_num_lcasses = len(self.idx_head)
        
        f1_macro_head = MultilabelF1Score(num_labels = head_num_lcasses, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_micro_head = MultilabelF1Score(num_labels = head_num_lcasses, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_weighted_head = MultilabelF1Score(num_labels = head_num_lcasses, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_none_head = MultilabelF1Score(num_labels = head_num_lcasses, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)
        
        mcc_head = MatthewsCorrCoef(task = 'multilabel', num_labels = head_num_lcasses, 
                               ignore_index = -1, validate_args=val_args).to(self.device)
        
        #=========================== Mid Label Metrics =====================
        mid_num_lcasses = len(self.idx_mid)
        
        f1_macro_mid= MultilabelF1Score(num_labels = mid_num_lcasses, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_micro_mid = MultilabelF1Score(num_labels = mid_num_lcasses, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_weighted_mid = MultilabelF1Score(num_labels = mid_num_lcasses, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_none_mid = MultilabelF1Score(num_labels = mid_num_lcasses, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)
        
        mcc_mid = MatthewsCorrCoef(task = 'multilabel', num_labels = mid_num_lcasses, 
                               ignore_index = -1, validate_args=val_args).to(self.device)
        
        
        #=========================== Tail Label Metrics =====================
        tail_num_lcasses = len(self.idx_tail)
        
        f1_macro_tail= MultilabelF1Score(num_labels = tail_num_lcasses, multidim_average = 'global',
                               average='macro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_micro_tail = MultilabelF1Score(num_labels = tail_num_lcasses, multidim_average = 'global',
                               average='micro', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_weighted_tail = MultilabelF1Score(num_labels = tail_num_lcasses, multidim_average = 'global',
                               average='weighted', ignore_index = -1, validate_args=val_args).to(self.device)
        
        f1_none_tail = MultilabelF1Score(num_labels = tail_num_lcasses, multidim_average = 'global',
                               average='none', ignore_index = -1, validate_args=val_args).to(self.device)
        
        mcc_tail = MatthewsCorrCoef(task = 'multilabel', num_labels = tail_num_lcasses, 
                               ignore_index = -1, validate_args=val_args).to(self.device)
        
        with torch.no_grad():
            
            for batch_idx, (inputs, labels, mask, ignored_label_idxs) in enumerate(test_loader):
                inputs = inputs.to(self.device, non_blocking = self.args.non_block)
                labels = labels.to(self.device, non_blocking = self.args.non_block)
                
                
                with autocast(enabled = self.args.use_amp):
                    if 'rbd_plm' in self.args.basemodel:
                        mask = self.replace_label_values(labels, -1,-1,-1)
                        mask = mask.to(self.device, non_blocking = self.args.non_block)
                        
                        ignored_label_idxs = (labels == -1)
                        ignored_label_idxs = ignored_label_idxs.to(self.device, non_blocking = self.args.non_block)
                        
                        pred, attns = self.get_predictions(self.model, inputs, mask = mask)                
                    else:
                        pred, attns = self.get_predictions(self.model, inputs)
                
                    if not is_test:
                        
                        if self.args.loss_fn !='bce':
                            loss = self.loss_fn(pred,labels, 
                                                rbd_plm_ignored_label_idxs = ignored_label_idxs,
                                                args = self.args)
                        else:
                            loss = self.loss_fn(pred,labels)
        
                        if self.args.loss_fn == 'bce':
                                unkown_label_mask = self.replace_label_values(labels, 0,1,1)
                                loss = (unkown_label_mask * loss)

                                if self.args.loss_reduce == 'mean':
                                    loss = loss.sum() / unkown_label_mask.sum()
                                elif self.args.loss_reduce == 'sum':
                                    loss = loss.sum()
                                
                        
                        test_loss += loss
                
                
                labels_long = labels.long()

                # to compute mcc per label
                if is_test and batch_idx == 0:
                    total_preds = pred.clone().detach().cpu().float()
                    total_labels = labels_long.clone().detach().cpu().float()
                elif is_test and batch_idx > 0:
                    total_preds = torch.cat((total_preds,
                                             pred.clone().detach().cpu().float()),
                                            dim = 0)
                    total_labels = torch.cat((total_labels,
                                              labels_long.clone().detach().cpu().float()),
                                             dim = 0)
                         
                #============ Full Data Metrics Per Batch =====================
                batch_mcc_mtr = mcc(pred,labels_long)
                
                batch_f1_mtr = f1_macro(pred,labels) 
                batch_f1_micro = f1_micro(pred,labels) 
                batch_f1_weighted = f1_weighted(pred,labels) 
                batch_f1_none = f1_none(pred,labels) 
                
                batch_hamming_mtr = hamming_macro(pred,labels) 
                batch_hamming_micro = hamming_micro(pred,labels) 
                batch_hamming_weighted = hamming_weighted(pred,labels) 
                batch_hamming_none = hamming_none(pred,labels) 

                batch_jaccard_macro = jaccard_macro(pred,labels_long)
                batch_jaccard_micro = jaccard_micro(pred,labels_long)
                batch_jaccard_weighted = jaccard_weighted(pred,labels_long)
                batch_jaccard_none = jaccard_none(pred,labels_long)

                
                #============ Head Label Metrics Per Batch =====================
                
                pred_head = pred[:,self.idx_head]
                labels_long_head = labels_long[:,self.idx_head]
                labels_head = labels[:,self.idx_head]
                
                batch_mcc_mtr_head = mcc_head(pred_head,labels_long_head)
                
                batch_f1_mtr_head = f1_macro_head(pred_head,labels_head) 
                batch_f1_micro_head = f1_micro_head(pred_head,labels_head) 
                batch_f1_weighted_head = f1_weighted_head(pred_head,labels_head) 
                batch_f1_none_head = f1_none_head(pred_head,labels_head) 
                
                #============ Mid Label Metrics Per Batch =====================
                pred_mid = pred[:,self.idx_mid]
                labels_mid_long = labels_long[:,self.idx_mid]
                labels_mid = labels[:,self.idx_mid]
                
                batch_mcc_mtr_mid = mcc_mid(pred_mid,labels_mid_long)
                
                batch_f1_mtr_mid = f1_macro_mid(pred_mid,labels_mid) 
                batch_f1_micro_mid = f1_micro_mid(pred_mid,labels_mid) 
                batch_f1_weighted_mid = f1_weighted_mid(pred_mid,labels_mid) 
                batch_f1_none_mid = f1_none_mid(pred_mid,labels_mid) 
                
                #============ Tail Label Metrics Per Batch =====================
                pred_tail = pred[:,self.idx_tail]
                labels_tail_long = labels_long[:,self.idx_tail]
                labels_tail = labels[:,self.idx_tail]
                
                batch_mcc_mtr_tail = mcc_tail(pred_tail,labels_tail_long)
                
                batch_f1_mtr_tail = f1_macro_tail(pred_tail,labels_tail) 
                batch_f1_micro_tail = f1_micro_tail(pred_tail,labels_tail) 
                batch_f1_weighted_tail = f1_weighted_tail(pred_tail,labels_tail) 
                batch_f1_none_tail = f1_none_tail(pred_tail,labels_tail) 
                
                
        metrics = {}
        
        #================ Full Data Metrics per Epoch =========================
        metrics['loss'] = test_loss
        
        metrics['mcc'] = mcc.compute()
        
        metrics['f1_macro'] = f1_macro.compute()
        metrics['f1_micro'] = f1_micro.compute()
        metrics['f1_weighted'] = f1_weighted.compute()
        metrics['f1_none'] = f1_none.compute()
        
        
        metrics['hamming_macro']= hamming_macro.compute()
        metrics['hamming_micro'] = hamming_micro.compute()
        metrics['hamming_weighted'] = hamming_weighted.compute()
        metrics['hamming_none'] = hamming_none.compute()
        
        metrics['jaccard_macro'] = jaccard_macro.compute()
        metrics['jaccard_micro'] = jaccard_micro.compute()
        metrics['jaccard_weighted'] = jaccard_weighted.compute()
        metrics['jaccard_none'] = jaccard_none.compute()
        
        #================ Head Label Metrics per Epoch =========================
        metrics['mcc_head'] = mcc_head.compute()
        
        metrics['f1_macro_head'] = f1_macro_head.compute()
        metrics['f1_micro_head'] = f1_micro_head.compute()
        metrics['f1_weighted_head'] = f1_weighted_head.compute()
        metrics['f1_none_head'] = f1_none_head.compute()
        
        #================ Mid Label Metrics per Epoch =========================
        metrics['mcc_mid'] = mcc_mid.compute()
        
        metrics['f1_macro_mid'] = f1_macro_mid.compute()
        metrics['f1_micro_mid'] = f1_micro_mid.compute()
        metrics['f1_weighted_mid'] = f1_weighted_mid.compute()
        metrics['f1_none_mid'] = f1_none_mid.compute()
        
        #================ Tail Label Metrics per Epoch =========================
        metrics['mcc_tail'] = mcc_tail.compute()
        
        metrics['f1_macro_tail'] = f1_macro_tail.compute()
        metrics['f1_micro_tail'] = f1_micro_tail.compute()
        metrics['f1_weighted_tail'] = f1_weighted_tail.compute()
        metrics['f1_none_tail'] = f1_none_tail.compute()

        #to compute mcc per label
        if is_test:
            mcc_none = binary_matthews_corrcoef(total_preds[:,0],
                                                total_labels[:,0],
                                                ignore_index = -1
                                                ).unsqueeze(dim = 0)
            for i in range(1,self.num_classes):
                tmp = binary_matthews_corrcoef(total_preds[:,i],
                                               total_labels[:,i],
                                               ignore_index = -1
                                               ).unsqueeze(dim = 0)
                mcc_none = torch.cat((mcc_none,tmp))   
            
            metrics['mcc_none'] = mcc_none

            logging.info(f'mcc_total: {metrics["mcc"]}')
            logging.info(f'mcc_head: {metrics["mcc_head"]}')
            logging.info(f'mcc_mid: {metrics["mcc_mid"]}')
            logging.info(f'mcc_tail: {metrics["mcc_tail"]}')
        else:
            metrics['mcc_none'] = 0
        
        print(f'\nVal set: Average loss: {test_loss}')
        
        logging.info(f'Epoch {epoch}')
        logging.info(f'\nVal set: Average loss: {test_loss}')
        logging.info('==========') 
        logging.info(f'Val MCC: {metrics["mcc"]}')
        
        if is_rank0_wandb and self.args.wandb_logging:
            if not is_test:
                wandb.log({'val/val_loss': test_loss, 
                           'val/val_mcc': metrics['mcc'],})
                wandb.log({'epoch': epoch, 
                       'val_loss': test_loss,
                       'val_mcc': metrics['mcc']})
            if is_test:
                wandb.log({'test/test_mcc': metrics['mcc'],
                           'test/epoch': epoch})
                wandb.log({'epoch': epoch, 
                       'test_mcc': metrics['mcc']})
        
        return metrics, attns
