import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import save_log, save_model, KNN_acc, linear_acc

def moco(f_q, f_k, train_loader, test_loader, pretrain_loader, optimizer, criterion, lr_scheduler, device, epoch_num=200, logdir='log_moco'):
    batch_size=256
    dim=128
    dict_size=16384
    m=0.999
    t=0.07

    queue = torch.randn(dict_size, dim).to(device) # (64*256, 128) 랜덤 초기화 (dict_size = 64 * batch_size)
    
    for q_param, k_param in zip(f_q.parameters(), f_k.parameters()):
        k_param.data.copy_(q_param)
        k_param.requires_grad = False

    best_knn_acc = 0.0
    writer = SummaryWriter(f'{logdir}')
    
    for epoch in range(epoch_num):

        f_q.train(True)
        running_loss = 0.0
        for inputs, _ in pretrain_loader:
            x_q, x_k = inputs
            x_q, x_k = x_q.to(device), x_k.to(device) # Positive Pair

            # 마지막 배치 크기: 80 (!= 256)
            current_batch_size = x_q.size(0)

            optimizer.zero_grad()
            
            q = f_q(x_q) # (batch size, dim) -> (256, 128), 128차원의 쿼리 256개
            k = f_k(x_k) # 128차원의 키 256개
            k = k.detach() # no gradient to keys

            # positive logits
            # batch matrix multiplication -> positive pair의 유사도 (q*k_+)
            l_pos = torch.bmm(q.view(current_batch_size, 1, dim), k.view(current_batch_size, dim, 1)) # (current_batch_size, 1, 1)

            # negative logits
            # matrix multiplication -> 쿼리 & 딕셔너리의 키값들과의 유사도 (q*k)
            l_neg = torch.matmul(q.view(current_batch_size, dim), queue.T) # (current_batch_size, dict_size) -> batch_size개의 쿼리와 dict_size개의 키끼리 유사도 (l_neg[i][j]: q_i * k_j)
            logits = torch.cat([l_pos.view(-1, 1), l_neg], dim=-1) # (current_batch_size, dict_size+1)

            labels = torch.zeros(current_batch_size, dtype=torch.long).to(device) # positive -> 1st column (idx=0)
            loss = criterion(logits/t, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Momentum update
            for q_param, k_param in zip(f_q.parameters(), f_k.parameters()):
                k_param.data.copy_(m*k_param + (1.0-m)*q_param)

            # queue.size(): (dict_size, dim)
            queue = torch.cat([queue[current_batch_size:, :], k])

        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Calculate KNN accuracy
        knn_acc = KNN_acc(f_q, train_loader, test_loader, device)
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        save_log(epoch, epoch_num, train_loss, knn_acc=knn_acc)
        if knn_acc > best_knn_acc:
            save_model(knn_acc, f_q, logdir, epoch)
            best_knn_acc = knn_acc
    linear_acc(f_q, epoch_num, 2048, 10, train_loader, test_loader, device)
    
    writer.close()
