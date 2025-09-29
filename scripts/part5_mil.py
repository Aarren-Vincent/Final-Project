# part5_mil.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MIL_FCNet(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, in_dim) -> returns (B,)
        return self.net(x).squeeze(-1)

def mil_ranking_loss(scores_pos, scores_neg, lambda_smooth=8e-5, lambda_sparsity=8e-5):
    # scores_pos, scores_neg: (batch, 32)
    max_pos = torch.max(scores_pos, dim=1)[0]
    max_neg = torch.max(scores_neg, dim=1)[0]
    min_pos = torch.min(scores_pos, dim=1)[0]
    loss1 = torch.mean(torch.clamp(1.0 - max_pos + max_neg, min=0.0))
    loss2 = torch.mean(torch.clamp(1.0 - max_pos + min_pos, min=0.0))
    smooth = torch.mean(torch.sum((scores_pos[:, :-1] - scores_pos[:, 1:])**2, dim=1))
    sparsity = torch.mean(torch.sum(scores_pos, dim=1))
    return loss1 + loss2 + lambda_smooth * smooth + lambda_sparsity * sparsity

def train_mil(model, dataloader, epochs=3, device='cpu'):
    opt = optim.Adagrad(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in tqdm(dataloader):
            # feats: (batch, 32, 2048)
            batch = feats.shape[0]
            if batch < 2:  # need pos+neg in one batch for the simplified trainer
                continue
            half = batch // 2
            feats_pos = feats[:half].to(device)
            feats_neg = feats[half:2*half].to(device)
            B, N, D = feats_pos.shape
            pos_flat = feats_pos.view(B*N, D)
            neg_flat = feats_neg.view(B*N, D)
            scores_pos = model(pos_flat).view(B, N)
            scores_neg = model(neg_flat).view(B, N)
            loss = mil_ranking_loss(scores_pos, scores_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(dataloader):.6f}")

def score_video_model(model, video_feats, device='cpu'):
    # video_feats: Tensor (32,2048) — return per-snippet scores
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = video_feats.to(device)  # (32,2048)
        scores = model(x)           # (32,)
    return scores.cpu().numpy()
