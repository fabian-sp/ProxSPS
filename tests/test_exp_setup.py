import numpy as np
import torch
import os
import sys
from torch.utils.data import DataLoader

sys.path.append('..')

import configs
from src import datasets
from src import models
from exp_main import run_exp

def test_matrix_fac1():
    """
    for all runs, train set should be identical
    """
    
    for exp_dict in configs.EXP_GROUPS['matrix_fac1']:
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        seed = 12345678 
        run_seed = seed + 2 + exp_dict.get('runs', 0) 
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
            
        train_set = datasets.get_dataset(dataset_name=exp_dict['dataset'],
                                         split='train',
                                         datadir='data',
                                         exp_dict=exp_dict,
                                         seed=seed)
        
        first_row = train_set.dataset.tensors[0][0,:]
        ref = torch.tensor([ 0.5537,  0.5013,  0.4693, -0.7453, -0.4953, -0.0243])
        assert torch.allclose(first_row, ref, rtol=1e-3, atol=1e-3)
        
        dl_gen = torch.Generator()
        dl_gen.manual_seed(run_seed) # seed should be same for each solver, but different over runs
    
        train_loader = DataLoader(train_set,
                                  drop_last=True,
                                  shuffle=True,
                                  generator=dl_gen,
                                  batch_size=exp_dict['batch_size'])
        
        batch = iter(train_loader).next()
        data, labels = batch["data"], batch["labels"]
        
        if exp_dict['runs'] == 0:
            ref = torch.tensor([ 0.0802,  1.4982,  0.4615,  0.0152, -0.2317,  0.6973])
            assert torch.allclose(data[0,:], ref, rtol=1e-3, atol=1e-3)
        elif exp_dict['runs'] == 1:
            ref = torch.tensor([ 1.9276,  0.8309,  2.1525,  1.8036, -0.7114,  0.4077])
            assert torch.allclose(data[0,:], ref, rtol=1e-3, atol=1e-3)
        
            
        # validation set has different seed for each run (for synthetic data)
        val_set = datasets.get_dataset(dataset_name=exp_dict['dataset'],
                                       split='val',
                                       datadir='data',
                                       exp_dict=exp_dict,
                                       seed=run_seed)
        
        # Load Model
        # ==================
        torch.manual_seed(seed)   
        torch.cuda.manual_seed_all(seed)
        base = models.Base(train_loader, exp_dict, device)
        
        # assert that always initialized identical
        ref = torch.tensor([0.3477,  0.3840,  0.1516, -0.3780,  0.2863, -0.2397])
        assert torch.allclose(base.model[0].weight[0,:], ref, rtol=1e-3, atol=1e-3)
    
    return



# test that there are no errors
def test_exp():
    run_exp('test')
    return 
