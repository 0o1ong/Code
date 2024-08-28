import torch
import torch.nn as nn
import torch.optim as optim
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

# negative cosine similarity
def D(p, z):
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean()

def simsiam(encoder, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, epoch_num, logdir):
    f = encoder # backbone + projection mlp
    h = Predictor().to(device) # prediction mlp
    pred_model = nn.Sequential(f, h) # encoder with predictor
    
    ###
    optimizer = optim.Adam(pred_model.parameters(), weight_decay=1e-6, lr=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-5)
    ###
    
    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    for epoch in range(epoch_num):
        pred_model.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            input1, input2 = inputs
            x1, x2 = input1.to(device), input2.to(device)

            optimizer.zero_grad()
            z1, z2 = f(x1), f(x2)
            p1, p2 = h(z1), h(z2)

            loss = D(p1, z2)/2 + D(p2, z1)/2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # KNN accuracy
        knn_acc = KNN_acc(f, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, pred_model, logdir, epoch)
            best_knn_acc = knn_acc

    writer.close()
