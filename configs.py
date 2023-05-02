from src.utils import cartesian_exp_group
import itertools 
import numpy as np

#%% define optimizers

def get_prox_sps_list(lr_list=[10.0, 1.0], fstar_list=[0.], lr_schedule_list=['sqrt']):
    prox_sps_list = []
    for lr, fstar, sched in itertools.product(lr_list, fstar_list, lr_schedule_list):
        prox_sps_list += [{'name':'prox-sps', 'lr': lr, 'fstar': fstar, 'lr_schedule': sched}]
    
    return prox_sps_list

def get_sps_list(lr_list=[10.0, 1.0], fstar_list=[0.], lr_schedule_list=['sqrt']):
    sps_list = []
    for lr, fstar, sched in itertools.product(lr_list, fstar_list, lr_schedule_list):
        sps_list += [{'name':'sps', 'lr': lr, 'fstar': fstar, 'lr_schedule': sched}]
    
    return sps_list

def get_sgd_list(lr_list=[10.0, 1.0], lr_schedule_list=['sqrt']):
    sgd_list = []
    for lr, sched in itertools.product(lr_list, lr_schedule_list):
        sgd_list += [{'name':'sgd', 'lr': lr, 'lr_schedule': sched}]
    
    return sgd_list


# ADAM
adamw_list = [{'name':'adamw', 'lr': 1e-3, 'lr_schedule': 'constant'}]


#%%

# define runs
run_list = list(range(10))

EXP_GROUPS = {}

########################################################################################
## IMAGENET
########################################################################################
EXP_GROUPS['imagenet32-resnet110-sgd'] = cartesian_exp_group({"dataset":["imagenet32"],  
                                            "model": "imagenet32-resnet110",
                                            "model_kwargs": {'use_bn': True},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"], 
                                            "acc_func":["softmax_accuracy"],  
                                            "opt": get_sgd_list(lr_list=[1.], lr_schedule_list=['sqrt']),
                                            "batch_size":[512],
                                            "max_epoch":[50],
                                            "runs": [1]})

EXP_GROUPS['imagenet32-resnet110-proxsps'] = cartesian_exp_group({"dataset":["imagenet32"],  
                                            "model": "imagenet32-resnet110",
                                            "model_kwargs": {'use_bn': True},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"], 
                                            "acc_func":["softmax_accuracy"],  
                                            "opt": get_prox_sps_list(lr_list=[1.], lr_schedule_list=['sqrt']),
                                            "batch_size":[512],
                                            "max_epoch":[50],
                                            "runs": [1]})

EXP_GROUPS['imagenet32-resnet110-sps'] = cartesian_exp_group({"dataset":["imagenet32"],  
                                            "model": "imagenet32-resnet110",
                                            "model_kwargs": {'use_bn': True},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"], 
                                            "acc_func":["softmax_accuracy"],  
                                            "opt": get_sps_list(lr_list=[1.], lr_schedule_list=['sqrt']),
                                            "batch_size":[512],
                                            "max_epoch":[50],
                                            "runs": [1]})

EXP_GROUPS['imagenet32-resnet110-adamw'] = cartesian_exp_group({"dataset":["imagenet32"],  
                                            "model": "imagenet32-resnet110",
                                            "model_kwargs": {'use_bn': True},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"], 
                                            "acc_func":["softmax_accuracy"],  
                                            "opt": adamw_list,
                                            "batch_size":[512],
                                            "max_epoch":[50],
                                            "runs": [1]})

########################################################################################
## CIFAR10
########################################################################################

EXP_GROUPS['cifar10-resnet56'] = cartesian_exp_group({"dataset":["cifar10"],
                                            "model": "resnet56",
                                            "model_kwargs": {'use_bn': False},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"],
                                            "acc_func":["softmax_accuracy"],
                                            "opt": get_prox_sps_list(lr_list=[1.], lr_schedule_list=['sqrt']) 
                                                    + get_sps_list(lr_list=[1.], lr_schedule_list=['sqrt'])
                                                    + adamw_list,
                                            "batch_size":[128],
                                            "max_epoch":[100],
                                            "runs": [0,1,2]})

EXP_GROUPS['cifar10-resnet110'] = cartesian_exp_group({"dataset":["cifar10"],
                                            "model": "resnet110",
                                            "model_kwargs": {'use_bn': False},
                                            "l2_lambda": [5e-6, 5e-5, 5e-4], 
                                            "loss_func": ["softmax_loss"],
                                            "acc_func":["softmax_accuracy"],
                                            "opt": get_prox_sps_list(lr_list=[1.], lr_schedule_list=['sqrt']) 
                                                    + get_sps_list(lr_list=[1.], lr_schedule_list=['sqrt'])
                                                    + adamw_list,
                                            "batch_size":[128],
                                            "max_epoch":[100],
                                            "runs": [0,1,2]})

########################################################################################
### MATRIX FACTORIZATION 
################################

EXP_GROUPS['matrix_fac1'] = cartesian_exp_group({"dataset": ['matrix_fac'],
                            "model": ['matrix_fac'],
                            "model_kwargs": {"rank": 4},
                            "p1": 6,
                            "p2": 10,
                            "n": 1000,
                            "cond": 1e-5,
                            "l2_lambda": [1e-5, 1e-3, 1e-4, 1e-2, 1e-1], 
                            "loss_func": ['squared_loss'],
                            "acc_func": ['squared_loss'],
                            "opt": get_prox_sps_list(lr_list=[10., 5., 1.], lr_schedule_list=['sqrt']) 
                                    + get_sps_list(lr_list=[10., 5., 1.], lr_schedule_list=['sqrt']),
                            "batch_size":[20],
                            "max_epoch":[50],
                            "runs": run_list})


EXP_GROUPS['matrix_fac1v2'] = cartesian_exp_group({"dataset": ['matrix_fac'],
                            "model": ['matrix_fac'],
                            "model_kwargs": {"rank": 4},
                            "p1": 6,
                            "p2": 10,
                            "n": 1000,
                            "cond": 1e-5,
                            "l2_lambda": [1e-3], 
                            "loss_func": ['squared_loss'],
                            "acc_func": ['squared_loss'],                
                            "opt": get_prox_sps_list(lr_list=np.linspace(2., 0.5, 5), lr_schedule_list=['sqrt', 'constant']) 
                                    + get_sps_list(lr_list=np.linspace(2., 0.5, 5), lr_schedule_list=['sqrt', 'constant'])
                                    + get_sgd_list(lr_list=np.linspace(0.7, 0.125, 5), lr_schedule_list=['sqrt', 'constant']),
                            "batch_size":[20],
                            "max_epoch":[50],
                            "runs": run_list})

EXP_GROUPS['matrix_fac2'] = cartesian_exp_group({"dataset": ['matrix_fac'],
                            "model": ['matrix_fac'],
                            "model_kwargs": {"rank": 10},
                            "p1": 6,
                            "p2": 10,
                            "n": 1000,
                            "noise": 0.05,
                            "cond": 1e-5,
                            "l2_lambda": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
                            "loss_func": ['squared_loss'],
                            "acc_func": ['squared_loss'],
                            "opt": get_prox_sps_list(lr_list=[10., 5., 1.], lr_schedule_list=['sqrt']) 
                                    + get_sps_list(lr_list=[10., 5., 1.], lr_schedule_list=['sqrt']),
                            "batch_size":[20],
                            "max_epoch":[50],
                            "runs": run_list})


EXP_GROUPS['matrix_fac2v2'] = cartesian_exp_group({"dataset": ['matrix_fac'],
                            "model": ['matrix_fac'],
                            "model_kwargs": {"rank": 10},
                            "p1": 6,
                            "p2": 10,
                            "n": 1000,
                            "noise": 0.05,
                            "cond": 1e-5,
                            "l2_lambda": [1e-3], 
                            "loss_func": ['squared_loss'],
                            "acc_func": ['squared_loss'],                
                            "opt": get_prox_sps_list(lr_list=np.linspace(2., 0.5, 5), lr_schedule_list=['sqrt', 'constant']) 
                                    + get_sps_list(lr_list=np.linspace(2., 0.5, 5), lr_schedule_list=['sqrt', 'constant'])
                                    + get_sgd_list(lr_list=np.linspace(0.7, 0.125, 5), lr_schedule_list=['sqrt', 'constant']),
                            "batch_size":[20],
                            "max_epoch":[50],
                            "runs": run_list})

EXP_GROUPS['matrix_fac3'] = cartesian_exp_group({"dataset": ['matrix_fac'],
                            "model": ['matrix_fac'],
                            "model_kwargs": {"rank": 3},
                            "p1": 6,
                            "p2": 10,
                            "n": 1000,
                            "cond": 1e-5,
                            "l2_lambda": [0], 
                            "loss_func": ['squared_loss'],
                            "acc_func": ['squared_loss'],                
                            "opt": get_sps_list(lr_list=np.linspace(2., 0.1, 6), lr_schedule_list=['sqrt', 'constant'])
                                    + get_sgd_list(lr_list=np.linspace(0.7, 0.125, 5), lr_schedule_list=['sqrt', 'constant']),
                            "batch_size":[20],
                            "max_epoch":[50],
                            "runs": run_list})

########################################################################################
### MATRIX COMPLETION
################################


EXP_GROUPS['sensor1'] = cartesian_exp_group({"dataset": ['sensor_data'],
                                "model": ['matrix_complete'],
                                "model_kwargs": {"rank": 24},
                                "l2_lambda": [5e-5,1e-4,5e-4], 
                                "loss_func": ['squared_loss'],
                                "acc_func": ['rmse'],                
                                "opt": get_prox_sps_list(lr_list=[10., 5.], lr_schedule_list=['constant']) +
                                        get_sps_list(lr_list=[10., 5.], lr_schedule_list=['constant']) + 
                                        get_sgd_list(lr_list=[5., 1.], lr_schedule_list=['constant'])+
                                        adamw_list,
                                "batch_size":[128],
                                "max_epoch":[100],
                                "runs": run_list})



########################################################################################
### ILLUSTRATIVE PURPOSES
################################

# logistic regression
EXP_GROUPS['test'] = cartesian_exp_group({"dataset": ['synthetic'],
                        "model": ['linear'],
                        "p": 20,
                        "n": 200,
                        "l2_lambda": [1e-3], 
                        "loss_func": ['logistic_loss'],
                        "acc_func": ['logistic_accuracy'],
                        "opt": get_prox_sps_list(lr_list=[10., 1.], lr_schedule_list=['constant'])
                                + get_sps_list(lr_list=[10., 1.], lr_schedule_list=['sqrt'])
                                + adamw_list,
                        "batch_size":[20],
                        "max_epoch":[7],
                        "runs": [0,1]})

