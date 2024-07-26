import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc, linear_acc

# NT-Xent loss function
def NT_Xent(z, temperature, device): # z.size(): (2batch_size, 512)
    z = F.normalize(z, dim=1)
    cos_sim = torch.matmul(z, z.T) / temperature
    cos_sim.fill_diagonal_(float('-inf')) # 같은 이미지에 대한 유사도(sim_{i, i}) 무시 가능
    batch_size = cos_sim.size(0)//2 # 각 Column마다 target은 positive pair (i <-> i+batch_size) 
    target = torch.cat([torch.arange(batch_size)+batch_size, torch.arange(batch_size)]).to(device)
    return F.cross_entropy(cos_sim, target)

def simclr(model, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, epoch_num, logdir):
    
    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    for epoch in range(epoch_num):
        model.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            inputs1, inputs2 = inputs
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

        # KNN accuracy
        knn_acc = KNN_acc(model, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        save_model(best_knn_acc, knn_acc, model, logdir, epoch)
    linear_acc(model, epoch_num, 2048, 10, train_loader, test_loader, device)
    writer.close()
