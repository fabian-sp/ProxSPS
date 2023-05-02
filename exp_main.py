"""
@author: Fabian Schaipp
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import configs
from src import models
from src import datasets
from src import utils as ut


def _run(exp_dict):
    print("########################################################################")
    
    # Load experiment config
    # ==================
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using GPU!")
    else:
        device = torch.device('cpu')
        
    # Set seed
    seed = 12345678 # base seed
    run_seed = seed + 2 + exp_dict.get('runs', 0) # run-dependent seed
    assert run_seed != seed
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    
    # Load data
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict['dataset'],
                                     split='train',
                                     datadir='data',
                                     exp_dict=exp_dict,
                                     seed=seed)
    
    dl_gen = torch.Generator()
    dl_gen.manual_seed(run_seed) # seed should be same for each solver, but different over runs

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              generator=dl_gen,
                              batch_size=exp_dict['batch_size'])
    
    
    
    # validation set has different seed for each run (for synthetic data)
    val_set = datasets.get_dataset(dataset_name=exp_dict['dataset'],
                                   split='val',
                                   datadir='data',
                                   exp_dict=exp_dict,
                                   seed=run_seed)
    
    # Load Model
    # ==================
   
    # reseed to have identical initial weights
    torch.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)
    base = models.Base(train_loader, exp_dict, device)
    
    # Compute optimal value
    # ==================
    if exp_dict['model'] == 'linear' and exp_dict['loss_func'] == 'logistic_loss':
        opt_val = ut.logreg_optimal_value(train_set, exp_dict)
        exp_dict['_opt_val'] = opt_val
    elif exp_dict['model'] == 'linear' and exp_dict['loss_func'] == 'squared_loss':
        opt_val = ut.ridge_optimal_value(train_set, exp_dict)
        exp_dict['_opt_val'] = opt_val
    
    # Testing
    # batch = iter(train_loader).next()
    # data, labels = batch["data"], batch["labels"]
    # base.model(data)
      
    # Initialize
    # ==================
    
    score_list = []
    
    for epoch in range(0, exp_dict['max_epoch']):
        
        # Record metrics
        score_dict = {'epoch': epoch}
        score_dict['lr'] = base.sched.get_last_lr()[0] # must be stored before sched.step()
                       
        # Train one epoch
        s_time = time.time()
        base.train_on_loader(train_loader)
        e_time = time.time()
    
        # Validate one epoch
        # train set
        if exp_dict['loss_func'] == exp_dict['acc_func']:
            train_dict = base.val_on_dataset(train_set, 
                                             metrics = [exp_dict['loss_func']],
                                             names = ['loss'])  
        else:
            train_dict = base.val_on_dataset(train_set, 
                                             metrics = [exp_dict['loss_func'], exp_dict['acc_func']],
                                             names = ['loss', 'score'])
        
        # validation set
        val_dict = base.val_on_dataset(val_set, metrics=[exp_dict['acc_func']],
                                       names=['score'])                       
               
        # Record more metrics
        score_dict.update(train_dict)
        score_dict.update(val_dict)
        score_dict['model_norm'] = base.get_l2_norm() # compute l2-norm of model params      
        score_dict['grad_norm'] = base.get_grad_norm() # norm of stochastic gradient
        score_dict['train_epoch_time'] = e_time - s_time       
        
        
        # === for matrix factorization/completion: store nuclear norm and rank ===
        if exp_dict['model'] == 'matrix_fac':
            with torch.no_grad():
                W = base.model[1].weight @ base.model[0].weight
        elif exp_dict['model'] == 'matrix_complete':
            with torch.no_grad():
                W = base.model.get_matrix()
        else:
            W = None
        
        if W is not None:
            score_dict['nuclear_norm'] = torch.linalg.norm(W, ord='nuc').item()
            score_dict['model_rank'] = torch.linalg.matrix_rank(W).item()
        # ===========================================
        
        # Store and reset step size list for SPS
        score_dict['step_size_list'] =  [float(np.format_float_scientific(t,5)) for t in base.opt.state['step_size_list']]
        
        if exp_dict['opt']['name'] in ['sps', 'prox-sps']:
            base.opt.state['step_size_list'] = list()

        # Add score_dict to score_list
        score_list += [score_dict]
        
    
    return score_list

#%%
"""
Structure of result

[ {'config': exp_dict, 'scores': score_list}, ....  
 ]

"""

def run_exp(exp_name, start=None, stop=None):
    print("Running experiment: %s" % exp_name)
    result_path = 'results/' + exp_name
    all_configs = configs.EXP_GROUPS[exp_name].copy()
    
    # determine which part should be run
    # if start not None, then load exisiting results
    if (start is not None) and (stop is None):
        configs_to_run = all_configs[start:]
        all_res = ut.load_json(result_path)
        assert len(all_res) == start, "Length of existing results and partial run does not match"
    elif (start is None) and (stop is not None):
        configs_to_run = all_configs[:stop]
        all_res = list()
    elif (start is not None) and (stop is not None):
        configs_to_run = all_configs[start:stop]
        all_res = ut.load_json(result_path)
        assert len(all_res) == start, "Length of existing results and partial run does not match"
    else:
        configs_to_run = all_configs.copy()
        all_res = list()
        
    # main loop
    for exp_dict in configs_to_run:
        this_config = exp_dict.copy()
        this_config['_start_time'] = str(datetime.datetime.now())
        score_list = _run(exp_dict) # run
        this_config['_end_time'] = str(datetime.datetime.now())
        
        all_res.append({'config': this_config, 'scores': score_list})
        ut.save_json(result_path, all_res) # store
    
    #model_path = 'results/models/' + exp_name
    #ut.torch_save(model_path, model.get_state_dict())
    print("This is the end!")
    return        
        

#%%
#exp_dict=configs.EXP_GROUPS[exp_list[0]][0]
exp_list = ['cifar10-resnet110']

if __name__ == '__main__':
    
    for exp_name in exp_list:
        run_exp(exp_name, start=None, stop=None)

        