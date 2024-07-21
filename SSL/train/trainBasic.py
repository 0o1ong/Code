import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging

def train_basic(model, train_loader, test_loader, epoch_num, learning_rate, logdir):
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(logdir, 'training.log')),
        logging.StreamHandler()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=learning_rate, dampening=False)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=learning_rate)
    writer = SummaryWriter(f'{logdir}')

    best_val_acc = 0.0

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                  milestones=[int(epoch_num * 0.5), int(epoch_num * 0.75)], 
                                                  gamma=0.1)

    for epoch in range(epoch_num):
        epoch_start_time = time.time()

        model.train(True)
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # eval
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)  # batchsize
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100
        val_loss /= len(test_loader)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        log_message = (f'Epoch [{epoch + 1}/{epoch_num}], Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, '
                       f'Time: {epoch_duration:.2f} seconds')
        logging.info(log_message)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1} with validation accuracy {val_acc:.2f}%')

    writer.close()
