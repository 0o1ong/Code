import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc

class Predictor(nn.Module):
    def __init__(self, in_dim, num_classes):
        self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, num_classes))
    def forward(self, x):
        return self.fc(x)

# negative cosine similarity
def neg_cos(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean()

def simsiam(model, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, epoch_num, logdir):
    pred_model = nn.Sequential(model, Predictor(512, 512).to(device)) # encoder with predictor
    
    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    for epoch in range(epoch_num):
        pred_model.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            input1, input2 = inputs
            input1, input2 = input1.to(device), input2.to(device)

            optimizer.zero_grad()
            
            pred1, pred2 = pred_model(input1), pred_model(input2)
            targ1, targ2 = model(input2).detach(), model(input1).detach()

            loss = neg_cos(pred1, targ1)/2 + neg_cos(pred2, targ2)/2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # KNN accuracy
        knn_acc = KNN_acc(model, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, pred_model, logdir, epoch)
            best_knn_acc = knn_acc

    writer.close()
