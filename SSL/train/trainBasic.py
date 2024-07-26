import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model

def train_basic(model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, epoch_num, logdir):

    best_val_acc = 0.0  
    writer = SummaryWriter(f'{logdir}')  
    for epoch in range(epoch_num):
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
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        save_log(epoch, epoch_num, train_loss, val_loss=val_loss, val_acc=val_acc)
        best_val_acc = save_model(best_val_acc, val_acc, model, logdir, epoch)
    writer.close()
