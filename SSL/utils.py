import os
import math
import logging
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def log_setting(logdir):
    if not os.path.exists(os.path.join(logdir)):
        os.makedirs(os.path.join(logdir))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler(os.path.join(logdir, 'training.log')),
                        logging.StreamHandler()
                        ])

class LinearWarmupCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, lr_lambda=self.schedule)

    def schedule(self, step):
        if step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            t = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))

class MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.fc(x)

class Module(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
    
    '''def get_module(self):
        return nn.Sequential(self.encoder, self.predictor)'''
    
    def cal_encoder(self, x):
        return self.encoder(x)
    
    def cal_predictor(self, x):
        return self.predictor(x)
    
    def forward(self, x):
        return self.cal_predictor(self.cal_encoder(x))
