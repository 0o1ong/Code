import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc

def moco(f_q, f_k, train_loader, test_loader, pretrain_loader, optimizer, criterion, lr_scheduler, device, epoch_num, logdir):
    dim=512
    dict_size=4096
    m=0.99
    t=0.1
    
    queue = torch.randn(dict_size, dim).to(device)
    queue.requires_grad = False
    queue = F.normalize(queue, dim=1)
    ptr = 0

    for q_param, k_param in zip(f_q.parameters(), f_k.parameters()):
        k_param.data.copy_(q_param.data)
        k_param.requires_grad = False

    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    
    for epoch in range(epoch_num):
        f_q.train(True)
        running_loss = 0.0

        for inputs, _ in pretrain_loader:
            x_q, x_k = inputs
            x_q, x_k = x_q.to(device), x_k.to(device) # Positive Pair
            current_batch_size = x_q.size(0)
            
            optimizer.zero_grad()
            
            q = f_q(x_q) # (batch size, dim)
            q = F.normalize(q, dim=1)
            k = f_k(x_k)
            k = F.normalize(k, dim=1)
            k = k.detach() # no gradient

            # positive logits
            # batch matrix multiplication -> positive pair의 유사도 (q*k_+)
            l_pos = torch.bmm(q.view(current_batch_size, 1, dim), k.view(current_batch_size, dim, 1)) # (current_batch_size, 1, 1)

            # negative logits
            # matrix multiplication -> 쿼리 & 딕셔너리의 키값들과의 유사도 (q*k)
            l_neg = torch.matmul(q.view(current_batch_size, dim), queue.T) # (current_batch_size, dict_size) -> batch_size개의 쿼리와 dict_size개의 키끼리 유사도 (l_neg[i][j]: q_i * k_j)
            logits = torch.cat([l_pos.view(current_batch_size, 1), l_neg], dim=-1) # (current_batch_size, dict_size+1)

            labels = torch.zeros(current_batch_size, dtype=torch.long).to(device) # positive -> 1st column (idx=0)
            loss = criterion(logits/t, labels)
            loss.backward()
            optimizer.step() # f_q update

            running_loss += loss.item()

            # Momentum update
            for q_param, k_param in zip(f_q.parameters(), f_k.parameters()):
                k_param.data.mul_(m).add_(q_param.data, alpha=1-m)

            # queue.size(): (dict_size, dim)
            queue[ptr:ptr+k.size()[0]] = k
            ptr = (ptr+k.size()[0]) % dict_size

        lr_scheduler.step()

        train_loss = running_loss / len(pretrain_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Calculate KNN accuracy
        knn_acc = KNN_acc(f_q, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, f_q, logdir, epoch)
            best_knn_acc = knn_acc

    writer.close()
