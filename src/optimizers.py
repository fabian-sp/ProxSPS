import math
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, _LRScheduler
import numpy as np
import warnings
import copy

from sps import SPS

from .adam_prox import ProxAdam

def get_optimizer(exp_dict, params):
    """
    Get the optimizer.
    """
    
    opt_dict = copy.deepcopy(exp_dict["opt"])
    opt_name = opt_dict['name']    
    weight_decay = exp_dict.get('l2_lambda', 0.)
    
    # (Prox)SPS
    if opt_name == 'prox-sps':
        opt = SPS(params, lr=opt_dict['lr'], weight_decay=weight_decay, fstar=opt_dict.get('fstar',0), prox=True)
            
    elif opt_name == 'sps':
        opt = SPS(params, lr=opt_dict['lr'], weight_decay=weight_decay, fstar=opt_dict.get('fstar',0), prox=False)
            
    # Adam      
    elif opt_name == 'adam':
        opt = torch.optim.Adam(params, lr=opt_dict['lr'], weight_decay=weight_decay)
        
    # for AdamW, use implementations from Orabona paper (Pytorch equivalent: set weight_decay=weight_decay/lr)
    elif opt_name == 'adamw':
        #opt = torch.optim.AdamW(params, lr=opt_dict['lr'], weight_decay=weight_decay)
        opt = ProxAdam(params, lr=opt_dict['lr'], weight_decay=weight_decay,
                       weight_decay_option='AdamW')
    # SGD
    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=opt_dict['lr'], weight_decay=weight_decay)

    elif opt_name == 'sgd-m':
        opt = torch.optim.SGD(params, lr=opt_dict['lr'], weight_decay=weight_decay, momentum=0.9)
           
    else:
        raise ValueError("opt %s does not exist..." % opt_name)
        
    print(opt)
    return opt



def get_scheduler(opt, opt_dict):
    
    name = opt_dict.get('lr_schedule', None)
    
    if (name is None) or (name == 'constant'):
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'exponential':
        scheduler = StepLR(opt, step_size=30, gamma=0.5)
     
    elif name == 'adaptive':
        scheduler = AdaptiveLR(opt)
    else:
        raise ValueError("Unknown larning rate schedule")
    
    return scheduler


# LR Scheduler for using adaptive alpha_k (only for SPS)
class AdaptiveLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False, default_lr=1.):
        self.opt = optimizer
        self.default_lr = default_lr
        super(AdaptiveLR, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        if self.opt.state.get('step_size_list'):
            if len(self.opt.state['step_size_list']) > 0:
                v = np.median(self.opt.state['step_size_list'])
            else:
                warnings.warn("Optimizer object has empty step_size_list, using 1.")
                v = self.default_lr # empty list
        else:
            warnings.warn("Optimizer object has no attribute step_size_list, using 1.")
            v = self.default_lr # no list attribute
            
        return [base_lr * v for base_lr in self.base_lrs]