import math
import torch
import torch.nn as nn

# Vision Transformer
class ViT(nn.Module):
    def __init__(self, in_dim=3, img_size=32, patch_size=4, d=384, num_layers=6, num_heads=6, num_classes=10):
        super().__init__()
        self.embedded_patch = PatchEmbedding(in_dim, patch_size, d) # (B, C, H, W) -> (B, N, D)
        num_patches = (img_size//patch_size)**2 # N
        self.cls_token = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, 1, d), std=0.02))
        self.pos_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, num_patches, d), std=0.02))
    
        self.layers = nn.ModuleList([TransformerEncoder(d=d, num_heads=num_heads) for _ in range(num_layers)]) # input & output size: (B, N, D)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d),
            # nn.Linear(d, num_classes)
            # Proj head
            nn.Linear(d, d//4),
            # nn.BatchNorm1d(d//4),
            nn.ReLU(), 
            nn.Linear(d//4, num_classes)
        )

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.embedded_patch(x) # (B, N, D)
        x += self.pos_embedding
        cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        z = torch.cat([cls_token, x], dim = 1) # (B, N+1, D)
        for layer in self.layers:
            z = layer(z)
        last_cls_token = z[:, 0] # z_L^0 (B, D)
        y = self.mlp_head(last_cls_token)
        return y # (B, num_classes)
    
# input image를 D차원의 벡터 N개로 (패치 임베딩)
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=3, patch_size=4, d=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, d, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2) # (B, D, N), N = HW/P^2 (패치 개수)
        x = x.transpose(1, 2) # (B, N, D) for transformer encoder input
        return x

# MultiHead Self Attention
class MSA(nn.Module):
    def __init__(self, d=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d // num_heads # head별 q, k, v 차원

        self.u_qkv = nn.Linear(d, 3 * self.d_h * self.num_heads) # [q, k, v] = zU_qkv
        self.u_msa = nn.Linear(self.num_heads * self.d_h, d) 

    def forward(self, x):
        B, N, _ = x.shape 
        qkv = self.u_qkv(x)  # (B, N, 3d_h * num_head)
        qkv = qkv.reshape(B, N, self.num_heads, 3 * self.d_h) # num_heads 차원 추가
        qkv = qkv.permute(0, 2, 1, 3) # (B, N, num_head, 3d_h) -> (B, num_head, N, 3d_h)
        q, k, v = torch.split(qkv, self.d_h, dim=-1)  # 각각 (B, num_head, N, d_h)

        qk = torch.matmul(q, torch.transpose(k, 3, 2)) / math.sqrt(self.d_h)  # (B, num_head, N, N)
        A = torch.softmax(qk, dim=-1)  # (B, num_head, N, N)
        SA = torch.matmul(A, v)  # (B, num_head, N, d_h)
        SA = SA.permute(0, 2, 1, 3) # (B, N, num_head, d_h)
        MSA = SA.reshape(B, N, self.num_heads * self.d_h) # (B, N, num_head * d_h)
        return self.u_msa(MSA)  # (B, N, D)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d=384, d_mlp=1536, num_heads=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.msa = MSA(d, num_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_mlp),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(d_mlp, d),
            # nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = self.msa(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x 
        return x
