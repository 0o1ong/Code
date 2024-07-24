import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import random
import logging
from .utils import KNN_acc, linear_acc

# v1 (Random Rotate)
def rotate_img_v1(images):
    rotations = [0, 90, 180, 270]
    rotated_images = []
    labels = []
    for img in images:
        label = random.choice(rotations) // 90 # 회전수 == 라벨
        rotated_images.append(torch.rot90(img, k=label, dims=[1, 2]))
        labels.append(label)
    return torch.stack(rotated_images), torch.tensor(labels)

# v2, v3 (4 Rotations) -> Batch 단위로 한번에 회전
def rotate_img_v2(images, angle):
    return torch.rot90(images, k=angle//90, dims=[2, 3])

def train_rotnet(model, train_loader, test_loader, epoch_num, learning_rate, logdir, version):

    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(logdir, version, 'training.log')),
        logging.StreamHandler()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, lr=learning_rate)
    writer = SummaryWriter(f'{logdir}/{version}')

    best_knn_acc = 0.0

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[30, 60, 80],
                                                  gamma=0.2)

    for epoch in range(epoch_num):
        epoch_start_time = time.time()

        model.train(True)
        running_loss = 0.0

        for inputs, _ in train_loader:
            if version == 'v1':
                inputs, labels = rotate_img_v1(inputs) # Random rotation 적용
            else:
                batch_size = inputs.size(0) # 128
                all_rotated_images = []
                all_labels = []

                for angle in [0, 90, 180, 270]:
                    rotated_images = rotate_img_v2(inputs, angle)
                    labels = torch.full((batch_size,), angle//90)
                    all_rotated_images.append(rotated_images)
                    all_labels.append(labels)
                
                inputs = torch.cat(all_rotated_images)
                labels = torch.cat(all_labels)

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
            for inputs, _ in test_loader:
                batch_size = inputs.size(0)
                if version == 'v1':
                    inputs, labels = rotate_img_v1(inputs)
                else:
                    all_rotated_images = []
                    all_labels = []

                    for angle in [0, 90, 180, 270]:
                        rotated_images = rotate_img_v2(inputs, angle)
                        labels = torch.full((batch_size,), angle//90)
                        all_rotated_images.append(rotated_images)
                        all_labels.append(labels)
                    
                    inputs = torch.cat(all_rotated_images)
                    labels = torch.cat(all_labels)
                
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.max(outputs, 1)[1]
                if version == 'v1':
                    total += batch_size
                else:
                    total += (batch_size * 4)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100
        val_loss /= len(test_loader)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # KNN accuracy
        knn_accuracy = KNN_acc(model, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_accuracy, epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        log_message = (f'Epoch [{epoch + 1}/{epoch_num}], Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, '
                       f'KNN Accuracy: {knn_accuracy:.2f}%, '
                       f'Time: {epoch_duration:.2f} seconds')
        logging.info(log_message)

        if knn_accuracy > best_knn_acc:
            best_knn_acc = knn_accuracy
            torch.save(model.state_dict(), os.path.join(logdir, version, 'best_model.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1} with KNN accuracy {knn_accuracy:.2f}%')
            
    # Last epoch: linear acc
    linear_accuracy = linear_acc(model, epoch, 512, 10, train_loader, test_loader, device)
    logging.info(f"Linear Accuracy: {linear_accuracy:.2f}%")

    writer.close()
