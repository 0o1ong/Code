import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging
from .utils import *
from collections import deque

# encoder: ResNet
    # output_dim(num_classes)=128 -> normalized by its L2-norm
    # BN 사용 X (shuffling BN)
# tau=0.07
# optimizer=SGD(weight_decay=1e-4, momentum=0.9)
# batch_size=256
# learning_rate=0.03
# epoch_num=200


def moco(encoder, train_loader, test_loader, epoch_num, learning_rate, logdir, dim=128, k=65536, m=0.999, t=0.07):
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(logdir, 'training.log')),
        logging.StreamHandler()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    f_q = encoder(num_classes=dim) # query encoder
    f_k = encoder(num_classes=dim) # key encoder

    queue = deque() # dict

    for q_param, k_param in zip(f_q.parameters(), f_k.parameters()):
        k_param = k_param.copy_(q_param)
        k_param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(f_q.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=learning_rate, dampening=False)
    writer = SummaryWriter(f'{logdir}')

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                  milestones=[int(epoch_num * 0.5), int(epoch_num * 0.75)], 
                                                  gamma=0.1)

    best_knn_acc = 0.0

    for epoch in range(epoch_num):
        epoch_start_time = time.time()

        f_q.train(True)
        running_loss = 0.0

        for x, _ in train_loader:
            x_q = aug(x) # (batch size, 3, 32, 32)
            x_k = aug(x) # (batch size, 3, 32, 32)
            x_q, x_k = x_q.to(device), x_k.to(device) # Positive Pair

            optimizer.zero_grad()
            
            q = f_q(x_q) # (batch size, dim) -> (256, 128), 128차원의 쿼리 256개
            k = f_k(x_k) # 128차원의 키 256개
            k = k.detach() # no gradient to keys

            # positive logits: N*1
            N = x.size()[0] # batch size
            l_pos = torch.bmm(q.view(N, 1, dim), k.view(N, dim, 1))

            # negative logits: N*K
            l_neg = torch.matmul(q.view(N, dim), )

            logits = torch.cat([l_pos, l_neg], dim=-1)

            labels = torch.zeros(N)
            loss = criterion(logits/t, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # f_k.parameters() = m*f_k.parameters() + (1-m)*f_q.parameters()

            # enqueue
            queue.append(k)
            # dequeue
            queue.popleft()

        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Calculate KNN accuracy
        knn_accuracy = KNN_acc(f_q, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_accuracy, epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        log_message = (f'Epoch [{epoch + 1}/{epoch_num}], Loss: {train_loss:.4f}, '
                       f'KNN Accuracy: {knn_accuracy:.2f}%, Time: {epoch_duration:.2f} seconds')
        logging.info(log_message)

        if knn_accuracy > best_knn_acc:
            best_knn_acc = knn_accuracy
            torch.save(f_q.state_dict(), os.path.join(logdir, 'best_model.pth'))
            logging.info(f'Checkpoint saved at epoch {epoch + 1} with KNN accuracy {knn_accuracy:.2f}%')
    
    linear_accuracy = linear_acc(f_q, epoch_num, train_loader, test_loader, device)
    logging.info(f"Linear Accuracy: {linear_accuracy:.2f}%")

    writer.close()
        
