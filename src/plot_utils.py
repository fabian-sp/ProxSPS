import itertools
import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt

def collect_configs(res):
    """
    collects all configurations of a run
    
    should be in sync with id_cols
    """
    all_config = dict()
    _def = {'lr': [], 'l2_lambda': [], 'lr_schedule': [],}
    
    for r in res:
        solver = r['config']['opt']['name']
        lr = r['config']['opt']['lr']
        lam = r['config']['l2_lambda']
        sched = r['config']['opt']['lr_schedule']
        
        if solver not in all_config.keys():
            empty = copy.deepcopy(_def)
            all_config[solver] = empty
        
        if lr not in all_config[solver]['lr']:
            all_config[solver]['lr'].append(lr)
        
        if lam not in all_config[solver]['l2_lambda']:
            all_config[solver]['l2_lambda'].append(lam)
        
        if sched not in all_config[solver]['lr_schedule']:
            all_config[solver]['lr_schedule'].append(sched)
        
    return all_config

def filter_configs(all_config, key='lr'):
    """
    collects all values of some of the config keys
    """
    vals = list(set(itertools.chain.from_iterable([v[key] for v in all_config.values()]))) 
    vals = np.array(vals)
    vals.sort()
    
    return list(vals)

def get_raw_df(res):
    """
    builds DataFrame with all experiment results
    """
    df_list = list()
    all_opt_val = list()
    
    for r in res:
        this_df = pd.DataFrame(r['scores'])
                            
        this_df['_solver'] = r['config']['opt']['name']
        this_df['_lr'] =  r['config']['opt']['lr']
        this_df['_run'] = r['config']['runs']
        this_df['_l2_lambda'] = r['config']['l2_lambda']
        this_df['_lr_schedule'] = r['config']['opt']['lr_schedule']
        
        # calc some new metrics
        this_df['reg'] = (r['config']['l2_lambda']/2) * this_df['model_norm']**2
        this_df['train_obj'] = this_df['train_loss'] + this_df['reg']
        
        # add opbjective gap
        if '_opt_val' in r['config'].keys():
            all_opt_val.append(r['config']['_opt_val'])
            this_df['opt_gap'] = this_df['train_obj'] - r['config']['_opt_val']
            assert np.all((this_df['opt_gap'].isna()) | (this_df['opt_gap'] >= -1e-10)), f"Found negative optimal gap {this_df['opt_gap'].min()}" 
        
        df_list.append(this_df)
        
    if len(all_opt_val) > 0:
        if max(all_opt_val) > min(all_opt_val) + 1e-10:
            print("WARNING: optimal values might be not identical!")
    
    df = pd.concat(df_list)   
    df = df.reset_index(drop=True)
    assert not df[['_solver', '_lr', '_l2_lambda', '_lr_schedule', 'epoch', '_run']].duplicated().any(), "We have duplicates!"
    return df
            
def get_base_df(raw_df, which='mean'):
    
    # determines what to group by (when aggregating over runs)
    id_cols = ['_solver', '_lr', '_l2_lambda', '_lr_schedule', 'epoch']
    
    if which=='mean':
        df = raw_df.groupby(id_cols, sort=False).mean().drop('_run',axis=1)
        df2 = raw_df.groupby(id_cols, sort=False).std().drop('_run',axis=1)
        df2.columns = [c+'_std' for c in df2.columns]
        df = pd.concat([df,df2], axis=1)
        
    elif which=='first':
        df = raw_df.groupby(id_cols, sort=False).first()
        assert len(df._run.unique()) == 1
        df = df.drop('_run',axis=1)
    
    else:
        raise KeyError("Unknown option")
        
    df = df.reset_index(level=-1) # moves epoch out of index    
    df['_id'] = df.index.to_numpy() # create id column
    df = df.reset_index() # single columns for each _id part (for filtering) 
    
    df = df.reset_index(drop=True) # reset index
    df.insert(0, '_id', df.pop('_id')) # move id column to front

    return df

#%% 
###################################################
##### PLOTTING
###################################################

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)


# aesthetics
color_dict = {'prox-sps': '#2C3E50','sps': '#E74C3C', 'sgd': '#3498DB',
              'adam': '#C4D6B0','adamw': '#C4D6B0', 'prox-adam': '#C4D6B0'}
zorder_dict = {'prox-sps': 5,'sps': 4, 'sgd': 3,'adam': 2, 'adamw': 2,'prox-adam': 2}

#C4D6B0
#AFC893

all_ls = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5))]
markevery_dict = {'sps': 5, 'prox-sps': 8, 'sgd': 10, 'adam': 12, 'adamw': 12, 'prox-adam': 14}

#E78F3C # from paletton matched to sps read
#ECF0F1  # grey
#marker_dict = {'sqrt': 'o', 'constant': '>', 'linear': '+', 'exponential':'p'}


# naming
def get_score_name(s, classify=True):
    
    if s=='train_obj': 
        return r'Objective $\psi(x^k)$'
    elif  s=='train_score':
        if classify:
            return 'Training Accuracy'
        else:
            return 'Training Error'
    elif s=='val_score':
        if classify:
            return 'Validation Accuracy'
        else:
            return 'Validation Error'                 
    elif s=='train_loss':
        return r'Loss $f(x^k)$'   
    elif s=='reg': 
        return r'Regularization $\varphi(x^k)$'
    elif s=='model_norm': 
        return r'$\|x^k\|$'
    elif s=='grad_norm':
        return r'$\|g_k\|$'
    elif s=='opt_gap':
        return r'$\psi(x^k)-\psi^\star$'
    elif s=='train_epoch_time':
        return 'Epoch training time [sec]'
    elif s=='nuclear_norm':
        return 'Nuclear norm'
    elif s=='model_rank':
        return 'Rank'
    else:
        raise KeyError("Unknown score name")
    return

def get_aes(r_dict, all_config):
    
    #col = color_dict[solver] 
    col = create_color(r_dict['lr_schedule'], all_config, r_dict['solver'])
    ls = all_ls[all_config[r_dict['solver']]['lr'].index(r_dict['lr'])]
    marker = 'o'
    
    return col, ls, marker    
    
# creates color from palette based on lr_schedule
def create_color(sched, all_config, solver, n_colors=5):
    all_sched = all_config[solver]['lr_schedule']
    pal = sns.light_palette(color_dict[solver], reverse=True, n_colors=n_colors)
    return pal[all_sched.index(sched)] 

###################################################
###################################################


def plot_metric(_s, df, all_config, sigma=0, log_scale=True, xlim=None, legend=True, classify=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4.5))
    else:
        fig = None
        
    counter = 0

    for r in df['_id'].unique():
        
        (solver, lr, lam, sched) = r
        r_dict = {'solver': solver, 'lr': lr, 'l2_lambda': lam, 'lr_schedule': sched}
        print(r_dict)
        
        this_df = df.loc[df._id==r, :]
    
        # aes are based on solver/lr/lam
        _col, _ls, _marker = get_aes(r_dict, all_config)
        
        if legend:
            label = solver + f', {sched}, ' 
            label += rf'$\alpha_0$={lr}' if solver != 'decsps' else rf'$1/c_0$={lr}'
        else:
            label = None
            
        if '_score' in _s:
            y = this_df[_s].rolling(5).median()
        else:
            y = this_df[_s]
            
        ax.plot(this_df.epoch, y, c=_col, ls=_ls, lw=1.5, marker=_marker, markersize=5, markevery=(markevery_dict[solver],40), #markeredgecolor = 'dimgrey', markeredgewidth=.3,\
                label=label, alpha=0.95, zorder=zorder_dict[solver])
        
        if sigma > 0:
            d = this_df[_s+'_std']
            ax.fill_between(this_df.epoch, y-sigma*d, y+sigma*d, color=_col, alpha=.15, zorder=-5)
    
        counter += 1
        
        
    if _s == 'opt_gap':
        ax.set_ylim(1e-4, 1e0)    
            
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(get_score_name(_s, classify))
    
    ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
    if log_scale or (_s=='opt_gap'):
        ax.set_yscale('log')
    
    return fig, ax



