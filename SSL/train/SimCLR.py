import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging
from ..trash.utils import KNN_acc, linear_acc

# NT-Xent loss function
def NT_Xent(z, temperature, device): # z.size(): (2batch_size, 512)
    z = F.normalize(z, dim=1)
    cos_sim = torch.matmul(z, z.T) / temperature
    cos_sim.fill_diagonal_(float('-inf')) # 같은 이미지에 대한 유사도(sim_{i, i}) 무시 가능
    batch_size = cos_sim.size(0) # 각 Column마다 target은 positive pair (i <-> i+batch_size) 
    target = torch.cat([torch.arange(batch_size)+batch_size, torch.arange(batch_size)]).to(device)
    return F.cross_entropy(cos_sim, target)

def simclr(model, train_loader, test_loader, pretrain_loader, epoch_num, learning_rate, logdir):
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(logdir, 'training.log')),
        logging.StreamHandler()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=learning_rate, dampening=False)
    writer = SummaryWriter(f'{logdir}')
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    best_knn_acc = 0.0

    for epoch in range(epoch_num):
        epoch_start_time = time.time()

        model.train(True)
        running_loss = 0.0

        for inputs1, inputs2, _ in pretrain_loader:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

            optimizer.zero_grad()
            out1 = model(inputs1)  # (batch_size, 512)
            out2 = model(inputs2)  # (batch_size, 512)
            z = torch.cat([out1, out2]) # i, i + batch_size -> positive pair, (2*batch_size, 512)
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
            logging.info(f'Checkpoint saved at epoch {epoch + 1} with KNN accuracy {knn_accuracy:.2f}%')
    
    linear_accuracy = linear_acc(model, epoch_num, 2048, 10, train_loader, test_loader, device)
    logging.info(f"Linear Accuracy: {linear_accuracy:.2f}%")

    writer.close()
