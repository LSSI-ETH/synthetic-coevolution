#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data.distributed
import torch.distributed as dist

from data_helpers.edit_distance_data_utils import batch_datasets_edit_distance
from data_helpers.main_data_utils import batch_datasets_main
from data_helpers.holdout_positions_data_utils import batch_datasets_holdout_positions

import logging
import datetime
import os

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
    elif 'holdout' in args.dataset:
        batch_datasets = batch_datasets_holdout_positions
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
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def save_checkpoint(model, optimizer, epoch, gpu, args, early_stopper,
                    loss = None, val = False, test = False):
    
    '''
    # save model checkpoint based on training, validation metrics, and testing status
    '''
    if val:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
    elif not val and test:
        chkpt_str = f'best_validation_model_final_{args.param_file}.pth'
    else:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
        
    patience_counter = early_stopper.counter
    early_stop_executed = early_stopper.early_stop_executed
    early_stop_flag_tensor = early_stopper.early_stop_flag_tensor

    if gpu == 0:
        print("epoch: {} ".format(epoch+1), flush = True)
        checkpointing_path = args.checkpoint_path + chkpt_str
        print("Saving the Checkpoint: {}".format(checkpointing_path), flush = True)
        chkpt_dict = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'patience_counter': patience_counter,
            'early_stop_executed': early_stop_executed,
            'early_stop_flag_tensor': early_stop_flag_tensor,
            }
        torch.save(chkpt_dict, checkpointing_path)
                
    elif torch.cuda.device_count() < 1:
        print("epoch: {} ".format(epoch+1))
        checkpointing_path = args.checkpoint_path + chkpt_str
        print("Saving the Checkpoint: {}".format(checkpointing_path))
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'patience_counter': patience_counter,
            'early_stop_executed': early_stop_executed,
            'early_stop_flag_tensor': early_stop_flag_tensor,
            }, checkpointing_path)

# for torch.compile model loading issue with torch 2.1.2
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

#https://github.com/pytorch/pytorch/issues/101107
def repair_checkpoint(ckpt, state_dict = 'model_state_dict', prefix_to_remove = 'module.'):
    #ckpt = torch.load(path)
    try:
        in_state_dict = ckpt[state_dict]
        pairings = [
        (src_key, remove_prefix(src_key, prefix_to_remove))
        for src_key in in_state_dict.keys()
        ]
        if all(src_key == dest_key for src_key, dest_key in pairings):
            return ckpt[state_dict]  # Do not write checkpoint if no need to repair!
        else:
            out_state_dict = {}
            for src_key, dest_key in pairings:
                out_state_dict[dest_key] = in_state_dict[src_key]
            ckpt[state_dict] = out_state_dict
        #return ckpt[state_dict]
        return ckpt
    except:
        print(f'Error: {state_dict} not found in checkpoint')
        print(f'Available keys: {ckpt.keys()}')
        print(f'ckpt: {ckpt}')
    
    

def load_checkpoint(model, optimizer, gpu, args, val = False, test = False):
    '''
    # load model checkpoint based on training, validation metrics, and testing status
    '''
    if val:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
    elif not val and test:
       chkpt_str = f'best_validation_model_final_{args.param_file}.pth'
    else:
        chkpt_str = f'checkpoint_best_validation_model_{args.param_file}.pth'
        
    print("--------------------------------------------")
    print("Checkpoint file found!")
    print("Loading Checkpoint From: {}".format(args.checkpoint_path + chkpt_str))
    
    if args.distributed:
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
        early_stop_executed = checkpoint['early_stop_executed']
        early_stop_flag_tensor = checkpoint['early_stop_flag_tensor']
    except:
        try:
            for prefix in ['_orig_mod.','module.']:
                checkpoint = repair_checkpoint(checkpoint,
                                            state_dict = 'model_state_dict',
                                            prefix_to_remove=prefix,
                                            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_number = checkpoint['epoch']
            loss = checkpoint['loss']
            patience_counter = checkpoint['patience_counter']
            early_stop_executed = checkpoint['early_stop_executed']
            early_stop_flag_tensor = checkpoint['early_stop_flag_tensor']

        except:
            print(f'Error: Repair_Checkpoint failed for checkpoint: {chkpt_str}')
            checkpoint = torch.load(args.checkpoint_path + chkpt_str, map_location=map_location)
        
            #try except accounts for case in which model is loaded:
            #1. in a DDP script prior to initial torch.DDP call
            #2. on CPU 
            try:
                print('Standard loading of checkpoint failed. Attempting to load checkpoint with prefix removal', flush = True)
                #https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841
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
                early_stop_executed = checkpoint['early_stop_executed']
                early_stop_flag_tensor = checkpoint['early_stop_flag_tensor']
            except:
                try:
                    print('Prefix removal loading of checkpoint failed. Attempting to load checkpoint with compiled prefix removal', flush = True)
                    print('Standard loading of checkpoint failed. Attempting to load checkpoint with _orig_mod.module prefix removal', flush = True)
                    print('\n')
                    print(f'RELOADING CHECKPIONT: {args.checkpoint_path + chkpt_str}', flush = True)

                    checkpoint = torch.load(args.checkpoint_path + chkpt_str, map_location=map_location)

                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k[17:] # remove 'module.' of DataParallel/DistributedDataParallel
                        new_state_dict[name] = v
                    checkpoint['model_state_dict'] = new_state_dict.copy()
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch_number = checkpoint['epoch']
                    loss = checkpoint['loss']
                    patience_counter = checkpoint['patience_counter']
                    early_stop_executed = checkpoint['early_stop_executed']
                    early_stop_flag_tensor = checkpoint['early_stop_flag_tensor']
                except:
                    
                    print('_orig_mod.module Prefix removal loading of checkpoint failed. Attempting to load checkpoint with compiled prefix removal', flush = True)
                    print('Standard loading of checkpoint failed. Attempting to load checkpoint with _orig_mod.module.module. prefix removal', flush = True)
                    print('\n')
                    print(f'RELOADING CHECKPIONT: {args.checkpoint_path + chkpt_str}', flush = True)

                    checkpoint = torch.load(args.checkpoint_path + chkpt_str, map_location=map_location)
                    
                    print(f'Final attempt Keys: {checkpoint.keys()}', flush = True)
                    print(f'Final attempt Keys: {checkpoint["model_state_dict"].keys()}', flush = True)
                    
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k[24:] # remove 'module.' of DataParallel/DistributedDataParallel
                        new_state_dict[name] = v
                    checkpoint['model_state_dict'] = new_state_dict.copy()
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch_number = checkpoint['epoch']
                    loss = checkpoint['loss']
                    patience_counter = checkpoint['patience_counter']
                    early_stop_executed = checkpoint['early_stop_executed']
                    early_stop_flag_tensor = checkpoint['early_stop_flag_tensor']
    

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[gpu],
                                                          output_device=gpu,
                                                          #find_unused_parameters=True
                                                         )
    loss = checkpoint['loss']
    print("Checkpoint File Loaded - epoch_number: {}".format(epoch_number))
    print('Resuming training from epoch: {}'.format(epoch_number+1))
    print("--------------------------------------------")

    early_stopper = EarlyStopper(patience=args.patience, device = gpu)
    early_stopper.min_validation_loss = loss
    early_stopper.counter = patience_counter
    early_stopper.early_stop_executed = early_stop_executed
    early_stopper.early_stop_flag_tensor = early_stop_flag_tensor

    return model, optimizer, epoch_number, early_stopper



def check_for_and_load_checkpoints_and_early_stopper(model, optimizer, gpu, args,):
    
    start_epoch, epoch = 0,0

    early_stopper = EarlyStopper(patience=args.patience, device = gpu)
    early_stopper.min_validation_loss = float('inf')
    early_stopper.counter = 0
       
    if torch.cuda.device_count() > 0:
        #check if fully trained model is avilable
        if os.path.isfile(args.checkpoint_path + f'best_validation_model_final_{args.param_file}.pth'):
            model, optimizer, start_epoch, early_stopper = load_checkpoint(model, 
                                                                              optimizer,
                                                                              gpu,
                                                                              args,
                                                                              val = False,
                                                                              test = True)
            epoch = args.epochs + 1
    
        #check if best validation checkpoint exists
        elif os.path.isfile(args.checkpoint_path + f'checkpoint_best_validation_model_{args.param_file}.pth'):
            model, optimizer, start_epoch, early_stopper = load_checkpoint(model, 
                                                                              optimizer,
                                                                              gpu,
                                                                              args,
                                                                              val = True)
            epoch = start_epoch
        
        #check if any checkpoint exists
        elif os.path.isfile(args.checkpoint_path + f'checkpoint_{args.param_file}.pth'):
            model, optimizer, start_epoch, early_stopper = load_checkpoint(model, 
                                                                              optimizer,
                                                                              gpu,
                                                                              args)
            epoch = start_epoch

    return model, optimizer, epoch, start_epoch, early_stopper



def load_best_chkpt_save_as_final_model(model, optimizer,eval_loss, epoch, gpu, args, early_stopper):
    
    if torch.cuda.device_count() > 0:
        
        model, optimizer, epoch, early_stopper = load_checkpoint(model, 
                                                                optimizer,
                                                                gpu,
                                                                args,
                                                                val = True)
        dist.barrier()
        
    save_checkpoint(model, optimizer, epoch, gpu, args, 
            early_stopper, eval_loss,
            val = False,
            test = True)
    
    return model, optimizer


# early stopper based on validation loss & patience
class EarlyStopper:
    def __init__(self, patience=1, device='cpu'):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop_executed = False
        self.early_stop_flag_tensor = torch.zeros(1).to(device)
        self.early_stop_flag_tensor = self.early_stop_flag_tensor.float()

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

            if args.distributed:
                if gpu == 0:   
                    save_checkpoint(model, optimizer, epoch, gpu, args, 
                                    self, #early stopper
                                    validation_loss, 
                                    val = True)
            else:
                save_checkpoint(model, optimizer, epoch, gpu, args, 
                                    self, #early stopper
                                    validation_loss,                                     
                                    val = True)
        else:
            self.counter += 1

            if args.distributed:
                if gpu == 0:
                    if self.counter >= self.patience:
                        print(f'epoch: {epoch}', flush = True)
                        print(f'patience: {self.patience}', flush = True)
                        print(f'counter: {self.counter}', flush = True)
                        print(f'min_validation_loss: {self.min_validation_loss}', flush = True)
                        print(f'validation_loss: {validation_loss}', flush = True)
                        self.early_stop_flag_tensor += 1.1
                        self.early_stop_flag_tensor = self.early_stop_flag_tensor
                        print(f'GPU 0 self.early_stop_flag_tensor: {self.early_stop_flag_tensor}', flush = True)
                
                if self.early_stop_flag_tensor > 1:
                    self.early_stop_executed = True
                
            else:
                if self.counter >= self.patience:
                        print(f'epoch: {epoch}')
                        print(f'patience: {self.patience}')
                        print(f'counter: {self.counter}')
                        print(f'min_validation_loss: {self.min_validation_loss}')
                        print(f'validation_loss: {validation_loss}')
                        self.early_stop_executed = True
                        self.early_stop_flag_tensor += 1
        
        return self.early_stop_executed
