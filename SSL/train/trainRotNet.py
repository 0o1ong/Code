import torch
from torch.utils.tensorboard import SummaryWriter

import random
from .utils import save_log, save_model, KNN_acc, linear_acc

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

def train_rotnet(model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, epoch_num, logdir, version):

    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}/{version}')
    for epoch in range(epoch_num):
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

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # KNN accuracy
        knn_acc = KNN_acc(model, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, val_loss=val_loss, val_acc=val_acc, knn_acc=knn_acc)
        save_model(best_knn_acc, knn_acc, model, logdir, epoch)
    # Last epoch: linear acc
    linear_acc(model, epoch, 512, 10, train_loader, test_loader, device)
    writer.close()
