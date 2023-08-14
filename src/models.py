import tqdm
import numpy as np
import torch

from . import base_models
from . import optimizers
from .metrics import get_metric_function

 
class Base():
    def __init__(self, train_loader, exp_dict, device):
        super().__init__()
        self.exp_dict = exp_dict
        self.device = device
        
        self.model = base_models.get_model(exp_dict, train_set=train_loader.dataset)
        
        print(self.model)
        
        # Load Optimizer
        self.model.to(device=self.device)
        self.opt = optimizers.get_optimizer(exp_dict=exp_dict,
                                            params=self.model.parameters(),                             
                                            )
        
        # Load LR scheduler
        self.sched = optimizers.get_scheduler(self.opt, exp_dict["opt"])
        
        return 
    
    def train_on_loader(self, train_loader):
                
        self.model.train()

        pbar = tqdm.tqdm(train_loader)
        for batch in pbar:
            score_dict = self.train_on_batch(batch)
            pbar.set_description(f'Training - {score_dict["train_loss"]:.3f}')
        
        print("Current learning rate", self.sched.get_last_lr()[0])
        
        # update learning rate             
        self.sched.step()
        
        return
    
    def train_on_batch(self, batch):
        self.opt.zero_grad()
        
        loss_function = get_metric_function(self.exp_dict['loss_func'])
        
        # get batch and compute model output
        data, labels = batch["data"].to(device=self.device), batch["labels"].to(device=self.device)
        out = self.model(data)
               
        # loss func
        closure = lambda : loss_function(out, labels, backwards=True)
        loss = self.opt.step(closure)
        
        return {'train_loss': float(loss)}
    
    def val_on_dataset(self, dataset, metrics, names):
        self.model.eval()
        
        # we compute several metrics in one pass
        assert len(set(metrics)) == len(metrics), "Duplicate metric functions. This can cause trouble."
        
        metric_functions = dict()
        for _met in metrics:
            metric_functions[_met] = get_metric_function(_met)
        
        loader = torch.utils.data.DataLoader(dataset, drop_last=False, batch_size=self.exp_dict['batch_size'])
        score = dict(zip(metrics, np.zeros(len(metrics))))
        
        pbar = tqdm.tqdm(loader)
        for batch in pbar:
            # get batch and compute model output
            data, labels = batch["data"].to(device=self.device), batch["labels"].to(device=self.device)
            out = self.model(data)
            
            # compute score
            for j, _met in enumerate(metrics):
                this_metric = metric_functions[_met]
                score[_met] += this_metric(out, labels).item() * data.shape[0] # metric is averaged over batch --> multiply with batch size
                
            pbar.set_description(f'Validating {dataset.split}')
        
        # from sum to average
        for _met in metrics:
            score[_met] = float(score[_met] / len(loader.dataset))
        
        name_list = [dataset.split + '_' + n for n in names]       
        return dict(zip(name_list, score.values()))
      
    
    def get_l2_norm(self):
        w = 0.
        for p in self.model.parameters():
            w += (p**2).sum()
        return torch.sqrt(w).item()
    
    def get_grad_norm(self):
        grad_norm = 0.
        
        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    raise KeyError("None gradient")
                grad_norm += torch.sum(torch.mul(p.grad, p.grad))

        return torch.sqrt(grad_norm).item()
        
    


        



