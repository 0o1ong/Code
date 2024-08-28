import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc

# negative cosine similarity
def D(p, z):
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean()

def simsiam(online, empty_target, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, epoch_num, logdir):
    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    for epoch in range(epoch_num):
        online.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            input1, input2 = inputs
            x1, x2 = input1.to(device), input2.to(device)

            optimizer.zero_grad()
            z1, z2 = online.encoder(x1), online.encoder(x2)
            p1, p2 = online.predictor(z1), online.predictor(z2)

            loss = D(p1, z2)/2 + D(p2, z1)/2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        lr_scheduler.step()

        train_loss = running_loss / len(pretrain_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # KNN accuracy
        knn_acc = KNN_acc(online.encoder, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, online, logdir, epoch)
            best_knn_acc = knn_acc

    writer.close()
