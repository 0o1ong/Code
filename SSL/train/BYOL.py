import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc

class Predictor(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.fc(x)

def mse_loss(x, y):
    x = F.normalize(x)
    y = F.normalize(y)
    return 2 - 2 * (x * y).sum(dim=-1).mean()

def byol(backbone, target, train_loader, test_loader, pretrain_loader, optimizer, criterion, lr_scheduler, device, epoch_num, logdir):
    tau=0.99
    online = nn.Sequential(backbone, Predictor().to(device)) # add prediction layer

    for online_param, target_param in zip(backbone.parameters(), target.parameters()):
        target_param.data.copy_(online_param.data)
        target_param.requires_grad = False
    
    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    
    for epoch in range(epoch_num):
        online.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            input1, input2 = inputs
            input1, input2 = input1.to(device), input2.to(device)
            
            optimizer.zero_grad()

            pred1, pred2 = online(input1), online(input2)
            targ1, targ2 = target(input2).detach(), target(input1).detach()

            loss = mse_loss(pred1, targ1) + mse_loss(pred2, targ2)
            loss.backward()
            optimizer.step() # online network update

            running_loss += loss.item()

            # EMA
            for online_param, target_param in zip(backbone.parameters(), target.parameters()):
                target_param.data.mul_(tau).add_(online_param.data, alpha=1-tau)

        lr_scheduler.step()

        train_loss = running_loss / len(pretrain_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Calculate KNN accuracy
        knn_acc = KNN_acc(backbone, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, online, logdir, epoch)
            best_knn_acc = knn_acc
    writer.close()
