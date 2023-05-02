import os
import urllib
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import pandas as pd
from sklearn.datasets import make_low_rank_matrix
from torch.utils.data import Dataset
from torch.testing import assert_allclose

from .imagenet32 import get_imagenet32

_BASE_SEED = 12345678

class DatasetWrapper:
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return {"data": data, 
                'labels': target, 
                'meta': {'indices': index}}


def get_dataset(dataset_name, split, datadir, exp_dict, seed=None):
    train_flag = True if split == 'train' else False
    
    #=========================================================
    ## SYNTHETIC
    #=========================================================
    
    if dataset_name == 'synthetic':
        
        p = exp_dict['p']
        n = exp_dict['n']
        noise = exp_dict.get('noise', 0.)
        
        bias = 1
        scaling = 10 
        sparsity = 30 
        solutionSparsity = 1. # w is dense if equal 1
              
        classify = (exp_dict['loss_func'] == 'logistic_loss')
        
        # A has different seed over train/test and diff. runs
        rng0 = np.random.RandomState(seed)
        
        # oracle has fixed seed
        rng1 = np.random.RandomState(_BASE_SEED)
        w = rng1.randn(p) * (rng1.rand(p) < solutionSparsity)
        assert np.abs(w).max() > 0 , "The synthetic data generator returns an oracle which is zero."
        
        # scale w to norm 1
        w = w/np.linalg.norm(w)
        
        # A should be different for train and test
        A = rng0.randn(n,p) + bias
        A = A.dot(np.diag(scaling*rng0.randn(p)))
        A = A * (rng0.rand(n,p) < (sparsity*np.log(n)/n))
        
        assert np.linalg.norm(A, axis=1).min() > 0 , "A has zero rows"
        
        column_norm = exp_dict.get('column_norm', 40)
        A = column_norm * A / np.linalg.norm(A, axis=0)[None,:].clip(min=1e-6) # scale A column-wise --> L_i are differrent
        
        if classify:
            b = 2*(A.dot(w) >= 0).astype(int) - 1.
            b = b * np.sign(rng0.rand(n)-noise)
            labels = np.unique(b)
            
            assert np.all(np.isin(b, [-1,1])), f"Sth went wrong with class labels, have {labels}"
            print(f"Labels (-1/1) with count {(b==-1).sum()}/{(b==1).sum()}.")
            L_coef = 0.25
            
        # regression
        else:
            b = A.dot(w) + noise*rng0.randn(n)            
            L_coef = 2.
        
        #_Lmax = L_coef*(np.linalg.norm(A, axis=1)**2).max()
        #_L = L_coef*(np.linalg.norm(A, axis=1)**2).mean()
        
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))
        
        
    #=========================================================
    ## MATRIX FACTORIZATION SYNTHETIC
    #=========================================================
    elif dataset_name == 'matrix_fac':
        
        A,B = generate_synthetic_matrix_factorization_data(p=exp_dict['p1'], 
                                                           q=exp_dict['p2'], 
                                                           n_samples=exp_dict['n'], 
                                                           noise=exp_dict.get('noise', 0),
                                                           condition_number=exp_dict.get('cond', 1e-5), 
                                                           split=split, 
                                                           seed=seed)
        
        
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(B))
    
    #=========================================================
    ## SENSOR DATA
    #=========================================================
    elif dataset_name == 'sensor_data':
        # adapted from: https://github.com/andresgiraldo3312/DMF/blob/main/DMF_1.ipynb        
        db = pd.read_csv(datadir + '/' + 'sensors_month1.csv', sep = ',', index_col = 0)
        X = np.asarray(db) 

        rows, cols = np.nonzero(X)       # Use only nonzero entries                                                               
        indices = np.zeros((len(rows),2))  # sensor, time (row, column)
        values = np.zeros(len(rows))
        
        indices[:,0] = rows
        indices[:,1] = cols
        values = X[rows,cols]
        
        # permute dataset with fixed seed
        rng0 = np.random.RandomState(_BASE_SEED)
        indices = rng0.permutation(indices)
        rng0 = np.random.RandomState(_BASE_SEED)
        values = rng0.permutation(values)
                                                                      
        N_train = int(0.8*len(rows))                                                                   
        n_sensors, n_time = X.shape
        
        # demean and scale (only with train set info)
        mean_ = values[0:N_train].mean()
        std_ = values[0:N_train].std()
        
        values =  (values-mean_)/std_ # standardize
        
        assert_allclose(torch.tensor(values[0:5]), torch.tensor([-0.6056, -0.4624, 0.2748, -0.5206, -0.6135]), rtol=1e-3, atol=1e-3)
        
        if train_flag:
            dataset = torch.utils.data.TensorDataset(torch.IntTensor(indices[0:N_train,:]), torch.Tensor(values[0:N_train, None])) 
        else:
            dataset = torch.utils.data.TensorDataset(torch.IntTensor(indices[N_train:,:]), torch.Tensor(values[N_train:, None])) 
        
        # store 
        dataset.dim1 = n_sensors
        dataset.dim2 = n_time
        dataset.mean_ = mean_
        dataset.std_ = std_
        
        if train_flag:
            dataset.true_matrix = (X-mean_)/std_

    #=========================================================
    ## MNIST
    #=========================================================
    
    elif dataset_name in  ["mnist", "mnist_images"]:
        transf = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                  ]
        
        
        # input shape (784,)
        if dataset_name == "mnist":
            transf.append(torchvision.transforms.Lambda(lambda x: x.view(-1).view(784)))
        # else: input shape (1,28,28)
        
        dataset = torchvision.datasets.MNIST(datadir, 
                                             train=train_flag,
                                             download=True,
                                             transform=torchvision.transforms.Compose(transf)
                                             )   
    
    elif dataset_name in  ["emnist", "emnist_images"]:
        transf = [torchvision.transforms.ToTensor()
                  ]
               
        # input shape (784,)
        if dataset_name == "emnist":
            transf.append(torchvision.transforms.Lambda(lambda x: x.view(-1).view(784)))
        # else: input shape (1,28,28)
        
        dataset = torchvision.datasets.EMNIST(datadir,
                                              split='balanced',
                                              train=train_flag,
                                              download=True,
                                              transform=torchvision.transforms.Compose(transf)
                                             )  
        
        # IF split='letters': emnist labels are from 1 to 26, need 0 to 25 for CrossEntropyLoss
        # dataset.targets = dataset.targets - 1 
    
    #=========================================================
    ## LIBSVM
    #=========================================================
    
    elif dataset_name in LIBSVM_DOWNLOAD_FN.keys():

        X, y = load_libsvm(dataset_name, data_dir=datadir)
        
        if np.all(np.isin(y, [0,1])):
            y = y*2 - 1 # go from 0,1 to -1,1
        
        if dataset_name == 'breast-cancer':
            y[y==2] = 1
            y[y==4] = -1
        
        labels = np.unique(y)
        assert np.all(np.isin(y, [-1,1])), f"Sth went wrong with class labels, have {labels}"
        
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=_BASE_SEED)
        X_train, X_test, Y_train, Y_test = splits

        if train_flag:
            # training set
            X_train = torch.FloatTensor(X_train.toarray())
            Y_train = torch.FloatTensor(Y_train)
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        else:
            # test set
            X_test = torch.FloatTensor(X_test.toarray())
            Y_test = torch.FloatTensor(Y_test)
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
            
    #=========================================================
    ## IMAGENET
    #=========================================================
    elif dataset_name == "imagenet32":          
        dataset = get_imagenet32(split, path=datadir)
    #=========================================================
    ## CIFAR
    ## see https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
    #=========================================================
 
    elif dataset_name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if train_flag:
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, 4),
                                                  transforms.ToTensor(),
                                                  normalize,
                                                  ])
            
            dataset = torchvision.datasets.CIFAR10(root=datadir, train=True, 
                                                   download=True,
                                                   transform=transform_train
                                                   )
        else:
            
            transform_val = transforms.Compose([transforms.ToTensor(),
                                                normalize,
                                                ])
            
            dataset = torchvision.datasets.CIFAR10(root=datadir, train=False, 
                                                   download=True, 
                                                   transform=transform_val
                                                   )
    elif dataset_name == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

        if train_flag:
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
            
            dataset = torchvision.datasets.CIFAR100(root=datadir, train=True, 
                                                    download=True,
                                                    transform=transform_train
                                                    )
        else:
        
            transform_val = transforms.Compose([transforms.ToTensor(),
                                                normalize,
                                                ])
        
            dataset = torchvision.datasets.CIFAR100(root=datadir, train=False, 
                                                    download=True, 
                                                    transform=transform_val
                                                    )

    else:
        raise KeyError("Not a known dataset option!")
    
    return DatasetWrapper(dataset, split=split)

# ===========================================================
# matrix factorization

def generate_synthetic_matrix_factorization_data(p=6, q=10, n_samples=1000, noise=0, condition_number=None, split='train', seed=None):
    """
    Generate a synthetic matrix factorization dataset:
    Adapted from: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    
    NOTE: we use logspace instead of linspace for the condition number.            
    """
    # Atrue always uses same seed
    # measurements X depend on seed (use different seed for each run in validation set)
    rng0 = np.random.RandomState(seed)
    rng1 = np.random.RandomState(_BASE_SEED)
            
    # this is the same as multiplying D@B where B is random and D is diagonal with values as below:
    Atrue = np.logspace(1, np.log10(condition_number), q).reshape(-1, 1) * rng1.rand(q, p)
    
    # perturb train set
    if split == 'train':
        E = noise * (rng1.rand(q, p)*2-1) # E is in noise*[-1,1]
        Atrue *= (1+E)

    # create data and targets
    X = rng0.randn(p, n_samples)
    Ytrue = Atrue.dot(X) 
    data = (X.T, Ytrue.T)

    return data

# ===========================================================
# Helpers

from sklearn.datasets import load_svmlight_file


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "a1a"  : "a1a",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a",
                      "phishing"   : "phishing",  
                      "breast-cancer": "breast-cancer_scale"
                      }




def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y
