#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data.distributed
import torch.distributed as dist

from data_helpers.edit_distance_data_utils import batch_datasets_edit_distance
from data_helpers.main_data_utils import batch_datasets_main

import logging
import datetime

#=================================================
def initialize_best_metrics_dict():

    ''' initialize metrics logging dict '''
    
    best_metrics_dict = {}
    best_metrics_dict['mcc' ] = 0
    best_metrics_dict['f1_macro'] = 0
    best_metrics_dict['f1_micro'] = 0
    best_metrics_dict['f1_weighted'] = 0
    best_metrics_dict['f1_none'] = 0
    best_metrics_dict['hamming_macro']= 0
    best_metrics_dict['hamming_micro'] = 0
    best_metrics_dict['hamming_weighted'] = 0
    best_metrics_dict['hamming_none'] = 0
    best_metrics_dict['jaccard_macro'] = 0
    best_metrics_dict['jaccard_micro'] = 0
    best_metrics_dict['jaccard_weighted'] = 0
    best_metrics_dict['jaccard_none'] = 0
    best_metrics_dict['loss'] = 1e23
    
    #Head Label Metrics
    best_metrics_dict['mcc_head'] = 0
    best_metrics_dict['f1_macro_head'] = 0
    best_metrics_dict['f1_micro_head'] = 0
    best_metrics_dict['f1_weighted_head'] = 0
    
    #Mid Label Metrics
    best_metrics_dict['mcc_mid'] = 0
    best_metrics_dict['f1_macro_mid'] = 0
    best_metrics_dict['f1_micro_mid'] = 0
    best_metrics_dict['f1_weighted_mid'] = 0

    #Tail Label Metrics
    best_metrics_dict['mcc_tail'] = 0
    best_metrics_dict['f1_macro_tail'] = 0
    best_metrics_dict['f1_micro_tail'] = 0
    best_metrics_dict['f1_weighted_tail'] = 0
    return best_metrics_dict

# batch datasets based on task
def batch_datasets(args):    
    if 'ed' in args.dataset:
            batch_datasets = batch_datasets_edit_distance
    else:
        batch_datasets = batch_datasets_main
    return batch_datasets(args)


#============== Divide Label Set into Head, Middle, Tail Labels ===============
def bin_labels_by_frequency(args):
    
    '''
    This function takes a single argument `args`, which is expected to have a 
    property called `class_freq` containing label counts per class. 
    The function then sorts the frequencies of each label in `args.class_freq` 
    into three bins. The three groups are returned as three separate lists.
    '''
    # Extract the frequency values from `args.class_freq`
    assert len(args.class_freq) == 39 
    freq = args.class_freq
    sorted_label_frequencies = sorted(freq)
    
    # Calculate the threshold values for the three groups: Head, Mid, Tail
    low_mid_threshold = sorted_label_frequencies[13]
    high_mid_threshold = sorted_label_frequencies[26]

    # Group the labels based on their frequency & obtain label number (idx)
    group_high_freq = [i for i, x in enumerate(freq) if x>=high_mid_threshold]
    group_mid_freq = [i for i, x in enumerate(freq) if x<high_mid_threshold and x>low_mid_threshold]
    group_tail = [i for i, x in enumerate(freq) if x<=low_mid_threshold]
    
    # Return the binned label indices separately
    return group_high_freq, group_mid_freq, group_tail

def initialize_logger(args):
    
        current_time = str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logname = f'log_files/{args.dataset}_{args.basemodel}_{args.rbd_plm_backbone}_{args.param_file}_{current_time}'
        logging.basicConfig(filename=f'{logname}.log', level=logging.DEBUG)
        logging.info(f'Initializing log file {logname}.log')
        logging.info('\n----------------\n')
        logging.info('\n----------------\n')

# slurm distributed training
# https://github.com/ShigekiKarita/pytorch-distributed-slurm-example/blob/master/main_distributed.py
def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def save_checkpoint(model, optimizer, epoch, gpu, args, loss = None, 
                    patience_counter = 0, val = False, test = False):
    
    '''
    # save model checkpoint based on training, validation metrics, and testing status
    '''
    if val:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
    elif not val and test:
        chkpt_str = f'best_validation_model_final_{args.param_file}.pth'
    else:
        chkpt_str = f'checkpoint_{args.param_file}.pth'
        
    if gpu == 0:
        logging.info("epoch: {} ".format(epoch+1))
        checkpointing_path = args.checkpoint_path + chkpt_str
        logging.info("Saving the Checkpoint: {}".format(checkpointing_path))
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'patience_counter': patience_counter,
            }, checkpointing_path)
        
    elif torch.cuda.device_count() < 1:
        
        logging.info("epoch: {} ".format(epoch+1))
        checkpointing_path = args.checkpoint_path + chkpt_str
        logging.info("Saving the Checkpoint: {}".format(checkpointing_path))
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'patience_counter': patience_counter,
            }, checkpointing_path)
        
        
def load_checkpoint(model, optimizer, gpu, args, val = False, test = False):
    '''
    # load model checkpoint based on training, validation metrics, and testing status
    '''
    if val:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
    if not val and test:
       chkpt_str = f'best_validation_model_final_{args.param_file}.pth'
            
    logging.info("--------------------------------------------")
    logging.info("Checkpoint file found!")
    logging.info("Loading Checkpoint From: {}".format(args.checkpoint_path + chkpt_str))
    
    if args.distributed:
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    elif torch.cuda.device_count() > 0:
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
        
    checkpoint = torch.load(args.checkpoint_path + chkpt_str, map_location=map_location)
    

    #try except accounts for case in which model is loaded:
    #1. in a DDP script prior to initial torch.DDP call
    #2. on CPU 
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        loss = checkpoint['loss']
        patience_counter = checkpoint['patience_counter']
        
    except:
        # https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        checkpoint['model_state_dict'] = new_state_dict.copy()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        loss = checkpoint['loss']
        patience_counter = checkpoint['patience_counter']
        
            
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[gpu],
                                                          #find_unused_parameters=True
                                                         )
    loss = checkpoint['loss']
    logging.info("Checkpoint File Loaded - epoch_number: {}".format(epoch_number))
    logging.info('Resuming training from epoch: {}'.format(epoch_number+1))
    logging.info("--------------------------------------------")
    return model, optimizer, epoch_number, loss, patience_counter


# early stopper based on validation loss & patience
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop_executed = False

    def early_stop(self, 
                   validation_loss, 
                   model, 
                   optimizer, 
                   epoch, 
                   gpu, 
                   args):
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            save_checkpoint(model, optimizer, epoch, gpu, args, 
                            validation_loss, 
                            patience_counter = self.counter,
                            val = True)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'epoch: {epoch}')
                print(f'patience: {self.patience}')
                print(f'counter: {self.counter}')
                print(f'min_validation_loss: {self.min_validation_loss}')
                print(f'validation_loss: {validation_loss}')
                self.early_stop_executed = True
                return True
        return False