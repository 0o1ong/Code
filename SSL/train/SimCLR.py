import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging
from .utils import *

def simclr(model, train_loader, test_loader, epoch_num, learning_rate, logdir):
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(logdir, 'training.log')),
        logging.StreamHandler()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=learning_rate, dampening=False)
    writer = SummaryWriter(f'{logdir}')

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                  milestones=[int(epoch_num * 0.5), int(epoch_num * 0.75)], 
                                                  gamma=0.1)

    best_knn_acc = 0.0

    for epoch in range(epoch_num):
        epoch_start_time = time.time()

        model.train(True)
        running_loss = 0.0

        for inputs, _ in train_loader:
            aug1 = aug(inputs)
            aug2 = aug(inputs) # Positive Pair
            aug1, aug2 = aug1.to(device), aug2.to(device)

            optimizer.zero_grad()
            out1 = model(aug1)  # (batch_size, 512)
            out2 = model(aug2)  # (batch_size, 512)
            z = []
            for i in range(out1.size()[0]):
                z.append(out1[i])
                z.append(out2[i])
            z = torch.stack(z) # 2k-1, 2k -> positive pair, (2*batch_size, 512)
            loss = NT_Xent(z=z, temperature=0.5, device=device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Calculate KNN accuracy
        knn_accuracy = KNN_acc(model, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_accuracy, epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        log_message = (f'Epoch [{epoch + 1}/{epoch_num}], Loss: {train_loss:.4f}, '
                       f'KNN Accuracy: {knn_accuracy:.2f}%, Time: {epoch_duration:.2f} seconds')
        logging.info(log_message)

        if knn_accuracy > best_knn_acc:
            best_knn_acc = knn_accuracy
            torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1} with KNN accuracy {knn_accuracy:.2f}%')
    
    linear_accuracy = linear_acc(model, epoch_num, 2048, 512, train_loader, test_loader, device)
    logging.info(f"Linear Accuracy: {linear_accuracy:.2f}%")

    writer.close()
