#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# load pLM from hugging face model cache
#path = os.environ['HOME']
#os.environ['HF_HOME']=f"{path}/.cache/huggingface"
#os.environ['TRANSFORMERS_CACHE']=f"{path}/.cache/huggingface/models"
#os.environ['TRANSFORMERS_OFFLINE']= '1'

from config_args import get_args
from utils import batch_datasets, initialize_logger, initialize_best_metrics_dict
from utils import bin_labels_by_frequency, find_free_port, save_checkpoint 
from utils import load_checkpoint, EarlyStopper
from data_helpers.general_data_utils import get_voc_data_loader, voc_metrics
from train import Trainer

import re
import time
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist
import numpy as np
import datetime
import logging
from socket import gethostname #slurm
import wandb
import pandas as pd

# https://github.com/pytorch/pytorch/issues/1355
torch.set_num_threads(5)   
    
def main(gpu, args):
    
    # set seeds
    seed_entry = args.seed
    torch.manual_seed(seed_entry)
    torch.cuda.manual_seed(seed_entry)
    np.random.seed(seed_entry)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    
    if args.basemodel == 'rbd_plm_lr':
        assert args.rbd_plm_backbone == 'esm_8m'
    
    if torch.cuda.is_available():
            use_cuda = True
            args.non_block = True
            args.use_amp = True
    else:
        use_cuda = False
        args.non_block = False
        args.use_amp = False
    
    if torch.cuda.device_count() > 1:
        args.distributed = True
        args.data_parallel = True
        args.backend = 'nccl'
        
        # ddp slurm
        # https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
        #rank          = int(os.environ["SLURM_PROCID"])
        #world_size    = int(os.environ["SLURM_NPROCS"])
        #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        #args.num_gpus = gpus_per_node

        # ddp no slurm
        rank = gpu
        world_size = args.num_gpus
        assert world_size == torch.cuda.device_count()

        if rank == 0 :
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            init_url = "tcp://{}:{}".format(ip, port)

        dist.init_process_group(backend=args.backend,rank=rank, world_size=world_size, init_method=init_url, ) #slurm
        # https://github.com/ShigekiKarita/pytorch-distributed-slurm-example/blob/master/main_distributed.py
    
        args.batch_size = args.num_gpus * args.batch_size
        args.learn_rate = args.num_gpus * args.learn_rate 

        logging.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))   
        
        torch.cuda.set_device(gpu)

    device = torch.device(gpu if use_cuda else "cpu")
    

    # initialize data loaders
    # extra loader contains either highly labeled or class 3 sequences
    train_loader, val_loader, test_loader, args.class_freq, train_sampler = batch_datasets(args = args)
    he_voc_data_loader = get_voc_data_loader(voc_str = 'he', args = args)
    taft_voc_data_loader = get_voc_data_loader(voc_str = 'taft', args = args)
    
    args.train_len = len(train_loader.dataset) #used for args.loss_fn = dbloss
    args.val_len = len(val_loader.dataset)
    args.test_len = len(test_loader.dataset)
    
    # bin labels by freq into high, mid, and tail buckets
    args.label_idx_high, args.label_idx_mid, args.label_idx_tail = bin_labels_by_frequency(args)
    
    hparams = vars(args)
    logging.info(args)
    
    current_date = str(datetime.datetime.now().strftime("%d-%m-%Y"))
    
    # wandb logging initialization
    is_rank0_wandb_logging = False
    if args.wandb_logging:
        if torch.cuda.device_count() > 1:
            if rank == 0 :
                run = wandb.init(
                # set the wandb project where this run will be logged
                project=f'{args.wandb_project}_{args.dataset}_{args.basemodel}_{args.rbd_plm_backbone}_{current_date}',
                settings=wandb.Settings(start_method="fork"),
                # track hyperparameters and run metadata
                config=args
                )
            
                is_rank0_wandb_logging = True
            else:
                is_rank0_wandb_logging = False
            
        else:
            run = wandb.init(
            # set the wandb project where this run will be logged
            project=f'{args.wandb_project}_{current_date}',
            # track hyperparameters and run metadata
            config=args
            )
            is_rank0_wandb_logging = True
            

    #===================================== Initialize Model =======================================================
    model = Trainer(args,device)
    model.model = model.model.to(device)
        
    # Check if checkpoints exists
    training_completed = False
    
    # check if trained model is avilable
    if os.path.isfile(args.checkpoint_path + f'best_validation_model_final_{args.param_file}.pth'):
        model.model, model.optimizer, start_epoch, checkpoint_val_loss, patience_counter = load_checkpoint(model.model, 
                                                                          model.optimizer,
                                                                          gpu,
                                                                          args,
                                                                          val = False,
                                                                          test = True)
        epoch = args.epochs + 1
        training_completed = True

    #check if best validation checkpoint exists
    elif os.path.isfile(args.checkpoint_path + f'checkpoint_best_validation_model_{args.param_file}.pth'):
        model.model, model.optimizer, start_epoch, checkpoint_val_loss, patience_counter = load_checkpoint(model.model, 
                                                                          model.optimizer,
                                                                          gpu,
                                                                          args,
                                                                          val = True)
        epoch = start_epoch
        if patience_counter >= args.patience:
            training_completed = True
        
    else:
        start_epoch, epoch, checkpoint_val_loss, patience_counter = 0,0, float('inf'), 0
    
    if torch.cuda.device_count() > 1:
        logging.info("Using", torch.cuda.device_count(), "GPUs!")
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[gpu], output_device=gpu)
    
    model.model = torch.compile(model.model)
        
    logging.info(f'\nNow Training with model {args.basemodel}')
    logging.info(f'learn rate, scheduler, optimizer:  {args.learn_rate}, {args.lr_scheduler}, {args.opt_id}')
    logging.info(f'loss function: {args.loss_fn}')

    #======================     Train & Eval Cycle          ===================================
    metrics_dict = initialize_best_metrics_dict()

    if training_completed: 
        total_training_time = 0.
        
    elif not training_completed:
        # account for async cuda operations if using gpu
        if torch.cuda.device_count() > 0:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            training_start_time = time.time()    
           
        #Train / Eval
        logging.info(f'Normal training now starting at start ep: {start_epoch}')
        
        early_stopper = EarlyStopper(patience=args.patience)
        early_stopper.min_validation_loss = checkpoint_val_loss
        early_stopper.counter = patience_counter
       
        for epoch in range(start_epoch, args.epochs):
            logging.info(f'EPOCH {epoch}')
            
            model.train_step(train_loader,epoch,is_rank0_wandb_logging,args.batch_size, train_sampler)
            
            if args.evaluate_valset and epoch % args.evaluate_valset_interval == 0:
        
                metrics_dict, attns = model.test_step(val_loader, epoch, is_rank0_wandb_logging)
                
                #chkpt lowest val loss & eval early stopping
    
                #if epoch > 10 and early_stopper.early_stop(validation_loss = metrics_dict['loss'],
                if epoch > 10 and early_stopper.early_stop(validation_loss = metrics_dict['loss'],
                                                          model = model.model,
                                                          optimizer = model.optimizer,
                                                          epoch = epoch,
                                                          gpu = gpu,
                                                          args = args):                
                    break
    
        # account for async cuda operations if using gpu
        if torch.cuda.device_count() > 0:
            end_time.record()
            torch.cuda.synchronize()
            total_training_time = start_time.elapsed_time(end_time)/10**3
        else:                
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time

        if not training_completed:            
            
            # attempt to load compiled model
            try:
                # load best model from early stopping checkpoint
                model.model, model.optimizer, epoch, metrics_dict['loss'], _ = load_checkpoint(model.model, 
                                                                        model.optimizer,
                                                                        gpu,
                                                                        args,
                                                                        val = True,
                                                                        test = False)
            
            # if unable to load torch.compiled model, reinitialize & recompile first
            except:
                model = Trainer(args,device)
                model.model = model.model.to(device)
                # load best model from early stopping checkpoint
                model.model, model.optimizer, epoch, metrics_dict['loss'], _ = load_checkpoint(model.model, 
                                                                        model.optimizer,
                                                                        gpu,
                                                                        args,
                                                                        val = True,
                                                                        test = False)
                model.model = torch.compile(model.model)
                

            # save final model at end of training
            if gpu == 0 or device.type == 'cpu':
                save_checkpoint(model.model, model.optimizer, epoch, gpu, args, 
                        metrics_dict['loss'],
                        patience_counter = early_stopper.counter,
                        val = False,
                        test = True)

    test_metrics_dict, _  = model.test_step(test_loader, epoch, is_rank0_wandb_logging, is_test = True)
        
    test_metrics_dict = {f'{k}_test': v for k, v in test_metrics_dict.items()}.copy() #append '_test' to dictionary string names

    # Predict on variants of concern
    he_metrics_dict = voc_metrics(he_voc_data_loader, model, gpu, args, 
                        voc_str = 'he', is_test = True)

    taft_metrics_dict = voc_metrics(taft_voc_data_loader, model, gpu, args, 
                        voc_str = 'taft', is_test = True)

    del model
    logging.info('Now Deleting Model & writing metrics')
    
    def clean_metric_output(metric_dict):

        for key, value in metric_dict.items():
            if 'epoch' not in key:
                try:
                    metric_dict[key] = metric_dict[key].cpu().detach()
                except:
                    pass

        for key, value in metric_dict.items():
            if 'epoch' not in key:
                m = re.search(r'\((.*)\)', str(value))
                try:
                    metric_dict[key] = float(m.group(1))
                except:
                    pass

        return metric_dict

    metrics_dict = clean_metric_output(metrics_dict)
    test_metrics_dict = clean_metric_output(test_metrics_dict)
    he_metrics_dict = clean_metric_output(he_metrics_dict)
    taft_metrics_dict = clean_metric_output(taft_metrics_dict)

    output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
    output_dict = {**hparams, **metrics_dict, **test_metrics_dict, 
               **he_metrics_dict, **taft_metrics_dict}
    output_dict['time'] = output_time
    output_dict['total_training_time'] = total_training_time

    output_path = args.output_data_dir
    filename = f'{output_path}/{args.dataset}_{args.basemodel}.csv'
    
    if gpu == 0 or device.type == 'cpu':
        df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', 
                                                                index=False, 
                                                                header=(not os.path.exists(filename)))
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    
    args = get_args()
    
    current_time = str(datetime.datetime.now().strftime("%d-%m-%Y"))
    args.param_file = f'{args.dataset}_{args.basemodel}_{args.rbd_plm_backbone}_{args.seed}_{current_time}'        
    args.wandb_project = args.param_file
    initialize_logger(args)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    if torch.cuda.device_count() > 1:
        
        # slurm job setup
        #param_folder = str(os.environ["PARAM_FOLDER"])
        #param_file_queue = int(os.environ["PARAM_FILE_QUEUE"])
        #slurm_arr_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        #slurm_job_id = int(os.environ["SLURM_JOBID"])
        #args.param_file = f'{param_folder}_{param_file_queue}h_{slurm_arr_id}'        
        #args.wandb_project = f'{param_folder}_slurm_job_id'
        #args.num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        
        args.num_gpus = torch.cuda.device_count()
        mp.spawn(main, nprocs=args.num_gpus, args=(args,))
    
    else:

        # slurm job setup 
        #param_folder = str(os.environ["PARAM_FOLDER"])
        #param_file_queue = int(os.environ["PARAM_FILE_QUEUE"])
        #slurm_arr_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        #slurm_job_id = int(os.environ["SLURM_JOBID"])
        #args.param_file = f'{param_folder}_{param_file_queue}h_{slurm_arr_id}'        
        #args.wandb_project = f'{param_folder}_slurm_job_id'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main(device, args)
    
    if args.wandb_logging:
        wandb.finish()