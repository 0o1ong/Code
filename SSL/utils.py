import os
import math
import logging
from torch.optim.lr_scheduler import LambdaLR

def log_setting(logdir):
    if not os.path.exists(os.path.join(logdir)):
        os.makedirs(os.path.join(logdir))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler(os.path.join(logdir, 'training.log')),
                        logging.StreamHandler()
                        ])

class LinearWarmupCosineAnnealingLR():
    def __init__(self, optimizer, max_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler = LambdaLR(optimizer, lr_lambda=self.schedule)

    def schedule(self, epoch):
        if epoch <= self.warmup_steps:
            return (epoch / self.warmup_steps) * self.max_lr
        else:
            t = (epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t)) * self.max_lr
    
    def step(self):
        self.scheduler.step()
