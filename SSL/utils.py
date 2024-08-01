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
