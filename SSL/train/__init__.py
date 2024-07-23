from .trainRotNet import *
from .trainBasic import *
from .SimCLR import *
from .utils import *

__all__ = ['train_rotnet', 'simclr', 'trainBasic', 'train']

def train(train_type, *args):
    if train_type == 'rotation':
        return train_rotnet(*args)
    elif train_type == 'simclr':
        return simclr(*args)
    elif train_type == 'basic':
        return train_basic(*args)
    else:
        raise ValueError(f"Unknown train type: {train_type}")
