import pickle
import json
import os
import itertools
import torch
import numpy as np

import copy


def load_json(fname, decode=None):
    with open(fname + ".json", "r") as json_file:
        d = json.load(json_file)
    return d

def save_json(fname, data):
    with open(fname+".json", "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
    return

def save_pkl(fname, data):
    """Save data in pkl format."""
    with open(fname+".pkl", "wb") as f:
        pickle.dump(data, f)
    return

def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname+".pkl", "rb") as f:
        return pickle.load(f)

# def torch_save(fname, obj):
#     """"Save data in torch format."""
#     # Define names of temporal files
#     fname_tmp = fname + ".tmp"
#     torch.save(obj, fname_tmp)
#     os.rename(fname_tmp, fname)
#     return



#%% computing optimal values with scikit learn

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

def logreg_optimal_value(train_set, exp_dict):
    
    X = train_set.dataset.tensors[0].numpy()
    y = train_set.dataset.tensors[1].numpy()
    
    n_samples = len(y)
    if exp_dict['l2_lambda'] > 0:
        penalty = 'l2'
        C = 1/(exp_dict['l2_lambda']*n_samples)
    else:
        penalty = 'none'
        C = 1/n_samples
    sk = LogisticRegression(penalty=penalty, tol=1e-10, C=C, fit_intercept=False, 
                            solver='lbfgs', max_iter=300, verbose=0)

    sk.fit(X,y)
    sol = sk.coef_[0,:]
    
    t1 = exp_dict['l2_lambda']/2 * np.linalg.norm(sol)**2
    t2 = np.log(1+np.exp(-y*(X@sol))).mean()
    
    return t1+t2


def ridge_optimal_value(train_set, exp_dict):
    
    X = train_set.dataset.tensors[0].numpy()
    y = train_set.dataset.tensors[1].numpy()
    
    n_samples = len(y)  
    al = n_samples * exp_dict['l2_lambda']/2
    
    sk = Ridge(alpha=al, tol=1e-10, fit_intercept=False, 
               solver='auto', max_iter=300)

    sk.fit(X,y)
    sol = sk.coef_
    
    t1 = exp_dict['l2_lambda']/2 * np.linalg.norm(sol)**2
    t2 = ((X@sol - y)**2).mean()
    
    return t1+t2

#%% copied from https://github.com/haven-ai/haven-ai/blob/master/haven/haven_utils/exp_utils.py


def cartesian_exp_group(exp_config, remove_none=False):
    """Cartesian experiment config.
    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.
    Parameters
    ----------
    exp_config : str
        Dictionary with the experiment Configuration
    Returns
    -------
    exp_list: str
        A list of experiments, each defines a single set of hyper-parameters
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), values)) for values in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        # remove hparams with None
        if remove_none:
            to_remove = []
            for k, v in exp_dict.items():
                if v is None:
                    to_remove += [k]
            for k in to_remove:
                del exp_dict[k]
        exp_list += [exp_dict]

    return exp_list



