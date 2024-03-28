#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description='Synthetic Coevolution RBD-pLM',
                                    fromfile_prefix_chars='@')

    # paths
    parser.add_argument('--train_path', type=str, default='data/')
    parser.add_argument('--output_data_dir', type=str, default='results/')
    parser.add_argument('--model_dir', type=str, default='model_weights/')
    parser.add_argument('--checkpoint_path', type=str, default='model_weights/')

    # hyperparameters
    
    # model
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--basemodel', default='rbd_plm', type=str,
                        help='base model',
                        choices = ['rbd_plm','cnn', 'transformer',
                                   'rbd_plm_lr', # "plm only" in paper
                                   'logistic_regression'])
    parser.add_argument('--dropout', default=0.1, type=float,
                    help='dropout fraction')
    
    # rbd-plm
    parser.add_argument('--hdim_expansion', default = 1, type = int,
                        help='transformer ffwd hidden dim expansion factor')
    parser.add_argument('--rbd_plm_backbone', default='emb', type = str,
                    help='rbd_plm backbone',
                    choices=['none', 'emb','esm_8m','esm_35m',
                         'esm_150m', 'esm_650m'])
    parser.add_argument('--return_attns', action='store_true',
                    help='return rbd_plm attention activations')
    parser.add_argument('--inter_attn_layers', default = 3, type = int,
                    help='number of inter attention layers')
    parser.add_argument('--combined_attn_layers', default = 1, type = int,
                    help='number of inter attention layers')
    parser.add_argument('--activation', default = 'relu', type = str,
                    choices=['relu','gelu'])    
    
    # masked label modeling
    parser.add_argument('--lmt_mask_fraction', default = 0.85, type = float,
                        help='fraction of labels to mask for rbd_plm')
    parser.add_argument('--lmt_threshold', default = 3, type = int,
                        help='minimum number of labels per seq required masked lable training')
    parser.add_argument('--initiate_lmt_threshold', default = 0.75, type = float,
                    help='threshold to activate lmt')    
    parser.add_argument('--use_lmt', action='store_false',
                        help='use lmt for rbd_plm')

    # esm params
    parser.add_argument('--return_esm_attns', action='store_true',
                        help='return esm attention tensors during fwd pass')
    
    # optimizer
    parser.add_argument('--opt_id', default='adam', type=str,
                        help='optimizer type',
                        choices = ['sgd', 'adam'])
    parser.add_argument('--learn_rate', default=3e-5, type=float,
                    help='initial optimizer learning rate')
    parser.add_argument('--weight_decay', default=1e-1, type=float,
                help='initial optimizer learning rate')
    parser.add_argument('--lr_scheduler', default=None, type=bool,
                    help='include learn rate scheduler')
    parser.add_argument('--bert_adam',  default = True, type=bool,
                    help='whether or not to use BertAdam. if True,'\
                        ' omit bias correction per'\
                            'https://arxiv.org/pdf/2006.05987.pdf')
    
    # epochs
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--patience', default=10, type=int,
                        help='number of training epochs')
    parser.add_argument('--evaluate_valset',  action='store_false',
                        help='whether or not to evaluate metrics on val set'\
                            ' during training. options: True, False')
    parser.add_argument('--evaluate_valset_interval', default = 1, type=int,
                        help='whether or not to evaluate metrics on val set during'\
                            ' training. options: True, False')
    
    # loss function
    parser.add_argument('--loss_fn', default='bce', type=str,
                        choices=['bce', 'focal_loss', 'dbloss'])
    parser.add_argument('--loss_reduce', default='mean', type=str,
                        help='BCEWithLogitLoss reduction, options none, mean, sum')    
    # data
    parser.add_argument('--dataset', default='ed_3', type=str,
                        choices=['ed_3','ed_10','main'])
    # gpu
    parser.add_argument('--distributed', default = False, type=bool,
                    help='use DistributedDataParallel')
    parser.add_argument('--backend', default = 'nccl', type=str,
                    help='ddp backeng (gloo, nccl, mpi, etc)')
    parser.add_argument('--non_block', default = False, type=bool,
                    help='ddp dataloader non_blocking')
    parser.add_argument('--use_amp', default = False, type=bool,
                    help='mixed percision training')
    parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
    
    # general training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grad_accumulation_steps', default=1, type = int,
                    help='gradient accumulation steps')
    parser.add_argument('--wandb_logging',action='store_false',
                        help='enable logging via wandb')
    
    parser.set_defaults()
    args = parser.parse_args()
    return args
    