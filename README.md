# Proximal Stochastic Polyak 

![](results/plots/misc/flow_squared.gif)

## About

We develop a proximal stochastic Polyak method `ProxSPS` for stochastic optimization. The main focus is how to handle regularization for adaptive methods like the Polyak step size. 

The methods `SPS` and `ProxSPS` from the paper are implemented in [`sps/sps.py`](sps/sps.py). If you want to use `ProxSPS`, make sure to set `prox=True`, for example

	from sps.sps import SPS
	SPS(params, lr=1, weight_decay=1e-3, prox=True)

## Experimental setup

The file `configs.py` contains all parameter configurations of the experiments. One or multiple experiments can be run with `exp_main.py` or with `run_exp.ipynb`.
Simply specify in the list the experiment ids from `configs.py` that you would like to run, for example `['matrix_fac1', 'cifar10-resnet110']`. Output is stored as a JSON file in the directory `output` and with the experiment id as filename.

The scripts automatically detects whether `cuda` is available and if so, runs on GPU.

## Comments

The starting point for this repository was the [offical SPS repository](https://github.com/IssamLaradji/sps). However, we carried out several refactoring steps and rewrote also slightly the optimizer in order to handle regularization.
