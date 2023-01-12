"""
@author: Fabian Schaipp
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from matplotlib.lines import Line2D


from src.utils import load_json
from src import plot_utils as put
from src.plot_utils import  color_dict, zorder_dict,  get_score_name

exp_name = 'cifar10-resnet110'

save = False # save plots or not
xlim = None
    
#%% load results

result_path = 'results/' + exp_name

res = load_json(result_path)

print(f"Loaded results for {len(res)} different configurations.")

all_config = put.collect_configs(res)
print(all_config)
all_solver = list(all_config.keys())

# classification or regression task
if res[0]['config']['loss_func'] in ['squared_loss']:
    classify = False
else:
    classify = True

# create DataFrames
raw_df = put.get_raw_df(res)
base_df = put.get_base_df(raw_df)

#%% plot metrics

df = put.get_base_df(raw_df)
all_l2 = put.filter_configs(all_config, 'l2_lambda')

def plot_metric(df, s, log_scale=False, sigma=0, save=False):

    if len(all_l2) > 1:
        figsize = (3*len(all_l2), 3.5)
    else:
        figsize = (6, 4.5)
    
    fig, axs = plt.subplots(1, len(all_l2), figsize=figsize)
    
    for j in range(len(all_l2)):
        ax = axs[j] if len(all_l2) > 1 else axs
        this_l2 = all_l2[j]
        
        df_sub = df[df._l2_lambda == this_l2] # filter on l2
        _, ax = put.plot_metric(s, df_sub, all_config, sigma=sigma, log_scale=log_scale, legend=(j+1==len(all_l2)), classify=classify, ax=ax)
        
        if j > 0:
            ax.set_ylabel('')
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=6)    
        
        if this_l2 > 0:
            ax.set_title(rf"$\lambda= {this_l2}$", fontsize=10)
        
        if xlim is not None:
            ax.set_xlim(0,xlim)
        else:
            ax.set_xlim(0,)
            
        if classify and s in ['val_score', 'train_score']:
            ax.set_ylim(0,1)
        
    _legend_loc = 'upper right'    
    fig.legend(loc=_legend_loc, fontsize=8, ncol=len(all_solver), framealpha=0.9)
            
    fig.tight_layout()
    fig.subplots_adjust(top=0.77)
    
    #if len(all_l2) > 1 :
    #    fig.subplots_adjust(wspace=0.2, left=0.060)
    
    if save:
        basedir = f'results/plots/{exp_name}/'
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        fig.savefig(basedir + s + '.pdf')
    
    return fig

################
fig = plot_metric(df, 'train_obj', log_scale=True, sigma=1, save=save)
fig = plot_metric(df, 'val_score', log_scale=False, sigma=1, save=save)
fig = plot_metric(df, 'model_norm', log_scale=False, sigma=1, save=save)

#%% plot error as function of regularization

df = put.get_base_df(raw_df)


def plot_path(df, s, which_epoch, window_size=10, sigma=0.5, log_scale=True, save=False):
    if which_epoch is None:
        which_epoch = df.epoch.max()
    
    df = df[(df.epoch <= which_epoch) & (df.epoch >= which_epoch - window_size)] # filter to relevant epochs
    
    df = df.groupby(['_solver', '_lr', '_l2_lambda', '_lr_schedule'])[[s, s+'_std']].median()
    df = df.reset_index(level='_l2_lambda') # move _l2_lambda outside of index
    df['_id'] = df.index.to_numpy() # id has no _l2_lambda here
    
    ################ PLOTTING #####################
    fig, ax = plt.subplots(figsize=(4/3*3.5,3.5))
    
    for r in df['_id'].unique():
            
        (solver, lr, sched) = r
        r_dict = {'solver': solver, 'lr': lr, 'lr_schedule': sched}
        print(r_dict)
        
        _col, _ls, _marker = put.get_aes(r_dict, all_config)
        this_df = df.loc[df._id==r, :].copy()
        
        _label = solver + f', {sched}, ' 
        _label += rf'$\alpha_0$={lr}' if solver != 'decsps' else rf'$1/c_0$={lr}'
        
        y = this_df[s]
        ax.plot(this_df._l2_lambda, y, c=_col, marker=_marker, ls=_ls, lw=1.5, alpha=0.95, zorder = zorder_dict[solver], label=_label)
        if sigma > 0:
            d = this_df[s+'_std']
            ax.fill_between(this_df._l2_lambda, y-sigma*d, y+sigma*d, color=_col, alpha=.15, zorder=-5)
            
        
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(get_score_name(s, classify))
        
    ax.grid(which='both', axis='y', lw=0.2, ls='--', zorder=-10)
    ax.set_xscale('log')
    if classify:
        ax.legend(fontsize=8, ncol=min(3,len(all_config.keys())), loc='lower right')
        #ax.set_ylim(0,1)
    else:
        ax.legend(fontsize=8, ncol=min(3,len(all_config.keys())), loc='upper right')
        
    if log_scale:
        ax.set_yscale('log')
    
    fig.tight_layout()
    
    if save:
        basedir = f'results/plots/{exp_name}/'
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        fig.savefig(basedir + 'lambda_path_' + s + '.pdf')
        
    return fig

################
fig = plot_path(df, s='val_score', which_epoch=xlim, window_size=10, sigma=0.5, log_scale=True, save=save)
fig = plot_path(df, s='model_norm', which_epoch=xlim, window_size=10, sigma=0.5, log_scale=False, save=save)

#%% plot stability analysis

df = put.get_base_df(raw_df)
all_sched = put.filter_configs(all_config, 'lr_schedule')

def plot_stability(df, s, relative=False, save=False):
    # plot distance to best observed
    if relative:
        assert s == 'train_obj', "relative plotting only for objective"
        min_val = df[s].min()
        print("Best value found: ", min_val)
        df[s] = df[s] - min_val
    
    fig, axs = plt.subplots(1, len(all_sched), figsize=(5*len(all_sched),4))
    solver_legend = list()
    
    for j in range(len(all_sched)):
        ax = axs[j] if len(all_sched) > 1 else axs
        _sched = all_sched[j]
        
        for _solver in all_solver:
            counter=0
            pal = sns.light_palette(put.color_dict[_solver], reverse=True, n_colors=len(all_config[_solver]['lr'])+1)
            solver_legend.append(Line2D([0], [0], color=put.color_dict[_solver], lw=3.5)) 
            
            for _lr in all_config[_solver]['lr']: 
                
                this_df = df.loc[(df._lr == _lr) & (df._lr_schedule == _sched) & (df._solver == _solver),]
                
                if len(this_df) == 0:
                    continue
                
                if  (j == 0) or (_solver=='decsps'):
                    label = f"${np.round(_lr,2)}$"
                else:
                    label = None
                
                ax.plot(this_df.epoch, this_df[s], c=pal[counter], ls='-', lw=2., marker='', markersize=5, markevery = (1,15),
                       alpha=1., label=label, zorder=zorder_dict[_solver])
                
                # for having lines with edge color
                ax.plot(this_df.epoch, this_df[s], c=pal[0], ls='-', lw=3.5, marker='', markersize=5, markevery = (1,15),
                        alpha=1., zorder=zorder_dict[_solver]-0.5)
                
                counter += 1
        
        ax.set_title(_sched, loc='left')
        ax.set_xlim(0,49)
        
        if relative:
            ax.set_ylim(1e-5, 1e-1)
                   
        ax.set_yscale('log')
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
        
        ax.set_xlabel('Epoch')
        if j == 0:
            _ylabel = put.get_score_name(s, classify=classify)
            if relative:
                _ylabel = r' $\psi(x^k) - \min_k \psi(x^k)$'
            ax.set_ylabel(_ylabel, fontsize=12)
        
    
    fig.legend(title=r"$\alpha_0$", title_fontsize=8, loc='upper right', fontsize=8, ncol=len(all_solver), framealpha=1)
    fig.legend(solver_legend, all_solver, loc='lower right', fontsize=8, framealpha=1)
    fig.tight_layout()
    
    if save:
        basedir = f'results/plots/{exp_name}/'
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        fig.savefig(basedir + 'stability_' +s + '.pdf')   

    return fig

################
fig = plot_stability(df, s='train_obj', relative=True, save=save)
fig = plot_stability(df, s='val_score', relative=False, save=save)

#%% plot step sizes
"""
For this plot, we do not want to use aggregated numbers, so only use
the first run
"""

# only takes the first run
df = put.get_base_df(raw_df, which='first')
df = df[df._solver.isin(['sps','prox-sps', 'decsps'])]
df = df.sort_values(['_l2_lambda', '_solver', '_lr'])

log_scale = True

if len(df['_l2_lambda'].unique()) > 1:
    ncol = len(df['_lr'].unique()) * len(df['_solver'].unique())
    nrow = len(df['_l2_lambda'].unique())
else:
    ncol = df[['_solver', '_lr']].value_counts().groupby('_solver').size().max()
    nrow = len(df['_solver'].unique())
    
counter = 0

fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*1.5))

for r in df['_id'].unique():
    
    (solver, lr, lam, sched) = r
    print(r)
    
    if solver not in ['sps','prox-sps', 'decsps']:
        continue
    
    
    this_df = df.loc[df._id==r, :]

    iter_per_epoch = len(this_df['step_size_list'].iloc[0])
    upsampled = np.linspace(this_df.epoch.values[0], this_df.epoch.values[-1],\
                            len(this_df)*iter_per_epoch)

    all_s = []
    all_s_median = []
    for j in this_df.index:
        all_s_median.append(np.median(this_df.loc[j,'step_size_list']))
        all_s += this_df.loc[j,'step_size_list'] 
    
    ax = axs.ravel()[counter]
    _c = color_dict[solver]
    
    label1 = r'$\alpha_k$' if solver != 'decsps' else r'$1/c_k$'
    ax.plot(this_df.epoch, this_df.lr, c='silver', lw=2.5, label=label1)
    
    label2 = r'median($\zeta_k$)' if solver != 'decsps' else r'$\hat{\gamma}_k$'    
    if solver in ['sps','prox-sps']:
        ax.scatter(upsampled, all_s, c=_c, s=5, alpha=0.35) #label=r'$\zeta_k$',
        ax.plot(this_df.epoch, all_s_median, c='gainsboro', marker='o', markevery=(5,7),\
                markerfacecolor=_c, markeredgecolor='gainsboro', lw=2.5, label=label2)
        
        ax.set_title(solver + ', ' + rf'$\alpha_0={lr}$' + ', ' + rf'$\lambda={lam}$', fontsize=8)    
    
    else:
        ax.plot(upsampled, all_s, c=_c, marker='o', markevery=(0,iter_per_epoch*10),\
                markerfacecolor=_c, markeredgecolor='darkgrey', lw=1., label=label2)
        
        ax.set_title(solver + ', ' + rf'$1/c_0={lr}$' + ', ' + rf'$\lambda={lam}$', fontsize=8)   
            
    ax.set_ylim(1e-3, 1e3)
    if xlim is not None:
        ax.set_xlim(0, xlim)
        
    if log_scale:
        ax.set_yscale('log')
    
    if counter%ncol == 0:
        ax.set_ylabel('Step size', fontsize=10)
        ax.tick_params(axis='y', which='major', labelsize=9)
        ax.tick_params(axis='y', which='minor', labelsize=6)    
    else:
        ax.set_yticks([])
        
    if counter >= ncol*(nrow-1):
        ax.set_xlabel('Epoch', fontsize=10)
        ax.tick_params(axis='x', which='both', labelsize=9)
    else:
        ax.set_xticks([])
    
    ax.legend(loc='upper right', fontsize=6)
    counter += 1

fig.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

if save:
    basedir = f'results/plots/{exp_name}/'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    fig.savefig(basedir + 'step_sizes.png', dpi=500)
