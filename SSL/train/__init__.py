from .trainRotNet import train_rotnet
from .trainBasic import train_basic
from .SimCLR import simclr
from .MoCo import moco
from .BYOL import byol
from .SimSiam import simsiam

__all__ = ['train_rotnet', 'simclr', 'moco', 'byol', 'simsiam', 'trainBasic', 'linear_acc', 'train']

def train(train_type, *args):
    if train_type == 'rotation':
        return train_rotnet(*args)
    elif train_type == 'simclr':
        return simclr(*args)
    elif train_type == 'moco':
        return moco(*args)
    elif train_type == 'byol':
        return byol(*args)
    elif train_type == 'simsiam':
        return simsiam(*args)
    elif train_type == 'basic':
        return train_basic(*args)
    else:
        raise ValueError(f"Unknown train type: {train_type}")
