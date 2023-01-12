import torch
from torch import nn
import math
import torchvision.models as torch_models

from .resnet import resnet32, resnet56, resnet110

def get_model(exp_dict, train_set):
    name = exp_dict['model']
    model_kwargs = exp_dict.get('model_kwargs', {})
    
    if name == "linear":
        batch = train_set[0]
        model = MLP(input_size=batch['data'].shape[0], output_size=1, hidden_sizes=[], bias=False, **model_kwargs)
    
    elif name == "matrix_fac":
        model = MatrixFac(input_size=exp_dict["p1"], output_size=exp_dict["p2"], rank=exp_dict["model_kwargs"]["rank"])
        
    elif name == "matrix_complete":
        model = MatrixComplete(dim1=train_set.dataset.dim1, dim2=train_set.dataset.dim2, rank=exp_dict["model_kwargs"]["rank"])
        
    elif name == "mlp":
        model = MLP(**model_kwargs)

    elif name == "convnet":
        model = ConvNet()

    elif name == "resnet32":
        model = resnet32(**model_kwargs)
    
    elif name == "resnet56":
        model = resnet56(**model_kwargs)
        
    elif name == "resnet110":
        model = resnet110(**model_kwargs)
    
    elif name == 'vgg13-cifar10':
        model =  vgg13(**model_kwargs)
    
    elif name == 'vgg19-cifar10':
        model =  vgg19(**model_kwargs)
    
    
    return model


# =====================================================
# MLP

def MLP(input_size=784, output_size=1, hidden_sizes=[512,512], bias=True, dropout=False):
    modules = []
    _hidden = hidden_sizes.copy()
    
    _hidden.insert(0, input_size)
    for i, layer in enumerate(_hidden[:-1]):
        modules.append(torch.nn.Linear(layer, _hidden[i+1],  bias=bias))

        modules.append(torch.nn.ReLU())
        if dropout:
            modules.append(torch.nn.Dropout(p=0.5))

    modules.append(torch.nn.Linear(_hidden[-1], output_size, bias=bias))

    return torch.nn.Sequential(*modules)

# =====================================================
# MATRIX FACTORIZATION

def MatrixFac(input_size, output_size, rank):
    
    W1 = torch.nn.Linear(input_size, rank, bias=False)
    W2 = torch.nn.Linear(rank, output_size, bias=False)
    
    return torch.nn.Sequential(W1, W2)
    
class MatrixComplete(nn.Module):
    def __init__(self, dim1, dim2, rank):
        super(MatrixComplete, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.U = torch.nn.Linear(dim1, rank, bias=False)
        self.V = torch.nn.Linear(dim2, rank, bias=False)
        
        # bias parameters (can be seen as rank-one matrices)
        self.bias_U = torch.nn.Parameter(torch.randn(dim1))
        self.bias_V = torch.nn.Parameter(torch.randn(dim2))
        
    def forward(self, x):
        # x contains [row of U, column of V]        
        x1 = torch.nn.functional.one_hot(x[:,0].long(), self.dim1).float()
        x2 = torch.nn.functional.one_hot(x[:,1].long(), self.dim2).float()
        
        prod = torch.diag(self.U(x1) @ self.V(x2).T).reshape(-1)
        b1 = x1 @ self.bias_U
        b2 = x2 @ self.bias_V
        
        return (prod + b1 + b2)[:,None] # [batch_size,1] output
    
    def get_matrix(self):
        W = self.U.weight.T @ self.V.weight + self.bias_U[:,None] + self.bias_V[None,:]
        return W
        

# =====================================================
# from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
# VGG architecture for CIFAR

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, use_bn=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if use_bn:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(use_bn=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], use_bn=use_bn))


def vgg13(use_bn=False):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], use_bn=use_bn))


def vgg16(use_bn=False):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], use_bn=use_bn))


def vgg19(use_bn=False):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], use_bn=use_bn))


# =====================================================
# Autoencoder for MNIST
# adapted from https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac


class Encoder(nn.Module):   
    def __init__(self, latent_dim=10, use_bn=False):
        super().__init__()
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d() if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, latent_dim=10, use_bn=False):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d() if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d() if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=10, use_bn=False):
        super().__init__()
        self.E = Encoder(latent_dim, use_bn)
        self.D = Decoder(latent_dim, use_bn)
    
    def forward(self, x):
        x = self.E(x)
        x = self.D(x)
        return x
        
        
    
# =====================================================
# ConvNet for MNIST


class ConvNet(nn.Module):
    def __init__(self):        
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        
        self.drop_out = nn.Dropout()
        # downsampling twice by factor 2 --> 7x7 output, 32 channels
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 10)
        
        return
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    