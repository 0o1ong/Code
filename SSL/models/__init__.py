from .LeNet import LeNet5
from .ResNet import ResNet, BasicBlock, PreActBlock, BottleNeck
from .RotNet import RotNet
from .ViT import ViT

__all__ = ['LeNet5', 'BasicBlock', 'PreActBlock', 'BottleNeck', 'ResNet', 'RotNet', 'ViT', 'get_model']

def get_model(model_name, *args, **kwargs):
    if model_name == 'lenet':
        return LeNet5(*args, **kwargs)
    elif model_name == 'resnet':
        return ResNet(*args, **kwargs)
    elif model_name == 'rotnet':
        return RotNet(*args, **kwargs)
    elif model_name == 'vit':
        return ViT(*args, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
