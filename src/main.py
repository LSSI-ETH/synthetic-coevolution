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
from utils import bin_labels_by_frequency, find_free_port
from utils import check_for_and_load_checkpoints_and_early_stopper
from utils import load_best_chkpt_save_as_final_model
from data_helpers.general_data_utils import get_voc_data_loader, voc_metrics, dataset_to_dataloader
from train import Trainer

import re
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import datetime
from socket import gethostname #slurm
import wandb
import pandas as pd

    
def main(gpu, args, train_dataset, val_dataset, test_dataset, class_freq, world_size = None, master_port = None):
    
    # set seeds
    seed_entry = args.seed
    torch.manual_seed(seed_entry)
    torch.cuda.manual_seed(seed_entry)
    np.random.seed(seed_entry)
    if args.distributed:
        torch.cuda.manual_seed_all(args.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    
    if 'rbd_plm' not in args.basemodel:
        args.rbd_plm_backbone = 'emb'
    
    #==========================================================================
    # Distributed Training & GPU Setup
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    else:
        rank = 0
        
    if torch.cuda.device_count() > 1:
        print('Entered DDP Logic in Main Loop', flush = True)
        args.distributed = True
        args.backend = 'nccl'
        
        #Torch DDP slurm initialization
        #https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
        os.environ['MASTER_PORT'] = master_port
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.num_gpus = gpus_per_node
        assert gpus_per_node == torch.cuda.device_count()

        dist.init_process_group(backend=args.backend,rank=gpu, world_size=args.num_gpus,)#slurm
        #https://github.com/ShigekiKarita/pytorch-distributed-slurm-example/blob/master/main_distributed.py
        
        args.batch_size = args.batch_size * args.num_gpus
        args.learn_rate = args.learn_rate * args.num_gpus

        rank = dist.get_rank()

        print('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))   
    #==========================================================================
        
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    
    # initialize data loaders if not already initialized
    if 'rbd_plm' not in args.basemodel:
        print('Now setting up dataset inside mp.spawn', flush = True)
        train_dataset, val_dataset, test_dataset, class_freq = batch_datasets(args)

    args.class_freq = class_freq

    train_loader, train_sampler = dataset_to_dataloader(train_dataset, is_test = False, args = args)
    val_loader, val_sampler = dataset_to_dataloader(val_dataset, is_test = False, args = args)
    test_loader, _ = dataset_to_dataloader(test_dataset, is_test = True, args = args)

    he_voc_data_loader = get_voc_data_loader(voc_str = 'he', args = args)
    taft_voc_data_loader = get_voc_data_loader(voc_str = 'taft', args = args)
    
    args.train_len = len(train_loader.dataset) #used for args.loss_fn = dbloss
    args.val_len = len(val_loader.dataset)
    args.test_len = len(test_loader.dataset)
    
    # bin labels by freq into high, mid, and tail buckets
    args.label_idx_high, args.label_idx_mid, args.label_idx_tail = bin_labels_by_frequency(args)
    
    hparams = vars(args)
    print(args, flush = True)
    
    current_date = str(datetime.datetime.now().strftime("%d-%m-%Y"))
    
    # wandb logging initialization
    is_rank0_wandb_logging = False
    if args.wandb_logging:
        print(f'Wandb Logging Enabled: {args.wandb_logging}', flush = True)
        print(f'torch.cuda.device_count() > 1: {torch.cuda.device_count() > 1}', flush = True)
        if torch.cuda.device_count() > 1:
            print(f'rank: {rank}', flush = True)
            if rank == 0 :
                print(f'rank: {rank} is logging', flush = True)
                run = wandb.init(
                # set the wandb project where this run will be logged
                project = f'{args.wandb_project}_{args.dataset}_{current_date}',
                settings=wandb.Settings(start_method="fork"),
                config=args
                )
            
                is_rank0_wandb_logging = True
            else:
                is_rank0_wandb_logging = False
            
        else:
            run = wandb.init(
            project=f'{args.wandb_project}_{current_date}',
            config=args
            )
            is_rank0_wandb_logging = True
            

    #===================================== Initialize Model =======================================================
    model = Trainer(args,device)
    model.model = model.model.to(device)
    
    print(f'model: {model.model}', flush = True)

    print(f'number params: {sum(p.numel() for p in model.model.parameters() if p.requires_grad)}', flush = True)
    
    model.model, model.optimizer, epoch, start_epoch, early_stopper = check_for_and_load_checkpoints_and_early_stopper(model.model, model.optimizer, gpu, args)
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!", flush = True)
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[gpu], output_device=gpu)
    
    # torch 2.1.2 cuda 12.1.1 issue 
    # https://discuss.pytorch.org/t/runtime-error-when-running-inference-on-a-compiled-nn-transformerencoder/198010
    if args.basemodel != 'transformer':
        model.model = torch.compile(model.model)
        
        
    print(f'\nNow Training with model {args.basemodel}', flush = True)
    print(f'dataset: {args.dataset}', flush = True)
    print(f'learn rate, scheduler, optimizer:  {args.learn_rate}, {args.lr_scheduler}, {args.opt_id}', flush = True)
    print(f'loss function: {args.loss_fn}', flush = True)
    

    #======================     Train & Eval Cycle          ===================================
    metrics_dict = initialize_best_metrics_dict()

    if early_stopper.early_stop_executed: 
        total_training_time = 0.
        
    elif not early_stopper.early_stop_executed:
        #account for async cuda operations if using gpu
        if torch.cuda.device_count() > 0:
            if rank == 0 :
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
        else:
            training_start_time = time.time()    
           
        #Train / Eval
        print(f'Normal training now starting at start ep: {start_epoch}', flush = True)
       
        for epoch in range(start_epoch, args.epochs):
            print(f'EPOCH {epoch}')
            
            if args.distributed:
                with model.model.join():
                    model.train_step(train_loader,epoch,is_rank0_wandb_logging,args.batch_size, train_sampler)
                torch.cuda.synchronize(device=gpu)
            else:
                model.train_step(train_loader,epoch,is_rank0_wandb_logging,args.batch_size, train_sampler)
            
            
            if args.evaluate_valset and epoch % args.evaluate_valset_interval == 0:
        
                metrics_dict, _ = model.test_step(val_loader, epoch, is_rank0_wandb_logging)
                
                if args.distributed: torch.cuda.synchronize(device=gpu)
                
                if epoch > args.warmup_epochs:
                    _ = early_stopper.early_stop(validation_loss = metrics_dict['loss'],
                                                        model = model.model,
                                                        optimizer = model.optimizer,
                                                        epoch = epoch,
                                                        gpu = gpu,
                                                        args = args)        
                    if args.distributed:
                        dist.broadcast(early_stopper.early_stop_flag_tensor, src=0)

                    if early_stopper.early_stop_flag_tensor >= 1:
                        print(f'Training early stopped on gpu {gpu}', flush = True)
                        
                        if args.distributed:
                            dist.barrier()
                        break

        if args.distributed:
            dist.barrier()
    
        
        if torch.cuda.device_count() > 0:
            if rank == 0 :
                end_time.record()
                torch.cuda.synchronize()
                total_training_time = start_time.elapsed_time(end_time)/10**3
        else:                
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
        
    # attempt to load compiled model
    print('Attempting to load best model from training run', flush = True)
    model_final = Trainer(args,gpu)
    model_final.model = model_final.model.to(gpu)
    model_final.model, model_final.optimizer = load_best_chkpt_save_as_final_model(model_final.model, model_final.optimizer,
                                                                                    metrics_dict['loss'], epoch, gpu, 
                                                                                    args, early_stopper)
    
    print(f'Final model loaded successfully to gpu: {gpu}', flush = True)
    if args.basemodel != 'transformer':
        model_final.model = torch.compile(model_final.model)
            
    test_metrics_dict, _  = model_final.test_step(test_loader, epoch, is_rank0_wandb_logging, is_test = True)
        
    test_metrics_dict = {f'{k}_test': v for k, v in test_metrics_dict.items()}.copy() #append '_test' to dictionary string names

    # Predict on variants of concern
    he_metrics_dict = voc_metrics(he_voc_data_loader, model, gpu, args, 
                        voc_str = 'he', is_test = True)

    taft_metrics_dict = voc_metrics(taft_voc_data_loader, model, gpu, args, 
                        voc_str = 'taft', is_test = True)

    del model
    del model_final
    print('Now Deleting Model & writing metrics', flush = True)
    
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
    try:
        output_dict['total_training_time'] = total_training_time
    except:
        output_dict['total_training_time'] = 'not available'

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

    if args.basemodel != 'transformer':
        torch.set_float32_matmul_precision('high') # torch compile
       
    # initiate datasets on master rank or cpu if possible
    # due to memory contraints, one-hot encoding datasets batched on all workers in mp.spawn
    if 'rbd_plm' in args.basemodel:
        train_dataset, val_dataset, test_dataset, class_freq = batch_datasets(args)
    else:
        train_dataset, val_dataset, test_dataset, class_freq = None, None, None, None
    
    args.non_block = False
    args.use_amp = False
    args.fused = False
    args.foreach = True

    if torch.cuda.device_count() > 0:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        args.non_block = True
        args.use_amp = True
        args.fused = True
        args.foreach = False
        
        if torch.cuda.device_count() > 1:
            args.distributed = True
            args.backend = 'nccl'
        
        # slurm & ddp launch        
        if args.run_location == 'slurm':
            param_folder = str(os.environ["PARAM_FOLDER"])
            slurm_arr_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            slurm_job_id = int(os.environ["SLURM_JOBID"])
            args.param_file = f'{param_folder}_{slurm_arr_id}'        
            args.num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        else:
            current_date = str(datetime.datetime.now().strftime("%d-%m-%Y"))
            param_folder = f'{args.dataset}_{args.model}_{current_date}'
            slurm_arr_id = ''
            slurm_job_id = 0
            args.num_gpus = torch.cuda.device_count()
            args.param_file = f'{param_folder}'

        os.environ["WANDB__SERVICE_WAIT"] = "300"
        initialize_logger(args)

        args.wandb_project = f'{param_folder}'
        
        # uncomment for ddp debug
        #os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        
        world_size = int(
            os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))

        master_port = find_free_port()
        print('Found Master Port. Now Spawning processes', flush = True)
        mp.spawn(main, nprocs=args.num_gpus, args=(args, train_dataset, val_dataset, test_dataset, class_freq, world_size, master_port))

    else:
        initialize_logger(args)
        os.environ["WANDB__SERVICE_WAIT"] = "300"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main(device, args, train_dataset, val_dataset, test_dataset, class_freq)
    
    if args.wandb_logging:
        wandb.finish()