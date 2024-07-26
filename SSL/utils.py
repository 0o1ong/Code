import os
import logging

def log_setting(logdir):
    if not os.path.exists(os.path.join(logdir)):
        os.makedirs(os.path.join(logdir))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler(os.path.join(logdir, 'training.log')),
                        logging.StreamHandler()
                        ])

class lr_scheduler():
    def __init__(self, warmup, decay, steps, min_lr, max_lr):
