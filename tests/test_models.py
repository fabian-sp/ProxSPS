import sys
sys.path.append('..')
from src.base_models import get_model

all_resnet = [{'model': 'resnet56', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': True}},
                {'model': 'resnet110', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': True}},
                {'model': 'resnet56', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': True}},
                {'model': 'resnet110', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': True}},
                {'model': 'resnet56', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': False}},
                {'model': 'resnet110', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': False}},
                {'model': 'resnet56', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': False}},
                {'model': 'resnet110', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': False}},
                ]
all_vgg = [{'model': 'vgg13', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': True}},
            {'model': 'vgg19', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': True}},
            {'model': 'vgg13', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': True}},
            {'model': 'vgg19', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': True}},
            {'model': 'vgg13', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': False}},
            {'model': 'vgg19', 'dataset': 'cifar10', 'model_kwargs': {'use_bn': False}},
            {'model': 'vgg13', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': False}},
            {'model': 'vgg19', 'dataset': 'cifar100', 'model_kwargs': {'use_bn': False}},
            ]


def test_resnet_models():
    for s in all_resnet:
        model = get_model(s, None)
        #print(model)
    return

def test_vgg_models():
    for s in all_vgg:
        model = get_model(s, None)
        #print(model)
    return