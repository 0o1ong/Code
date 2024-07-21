from .trainRotNet import *
from .trainBasic import *
from .utils import *

__all__ = ['train_rotnet', 'trainBasic', 'train']

def train(train_type, *args):
    if train_type == 'rotation':
        return train_rotnet(*args)
    elif train_type == 'basic':
        return train_basic(*args)
    else:
        raise ValueError(f"Unknown model: {train_type}")
