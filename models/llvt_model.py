#!/usr/bin/env python3
"""
Line-Level Vulnerability Tagger (LLVT).

Treats vulnerability localization as a SEQUENCE LABELING task:
each line gets a direct vulnerable/safe prediction, trained with
focal loss and noise-aware confidence weighting.

Optionally includes contrastive loss to push vulnerable line
embeddings away from safe line embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    Down-weights easy negatives to focus on hard examples.
    Critical for extreme class imbalance (99%+ safe lines).
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for line embeddings.
    Pulls vulnerable lines together, pushes them away from safe lines.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, mask=None):
        """
        Args:
            features: [N, D] normalized line embeddings
            labels: [N] binary (0=safe, 1=vulnerable)
            mask: [N] valid line mask
        """
        if mask is not None:
            valid = mask > 0
            features = features[valid]
            labels = labels[valid]

        if features.shape[0] < 2 or labels.sum() < 1 or (1 - labels).sum() < 1:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=1)
        N = features.shape[0]

        # Similarity matrix
        sim = torch.mm(features, features.t()) / self.temperature  # [N, N]

        # Same-class mask
        labels = labels.view(-1, 1)
        same_class = (labels == labels.t()).float()  # [N, N]

        # Exclude self
        eye = torch.eye(N, device=features.device)
        same_class = same_class - eye
        valid_pairs = 1.0 - eye

        # Log-sum-exp for stability
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        exp_logits = torch.exp(logits) * valid_pairs
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-prob over positive pairs
        pos_count = same_class.sum(dim=1)
        pos_count = pos_count.clamp(min=1)
        mean_log_prob = (same_class * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob.mean()
        return loss


class LinePooler(nn.Module):
    """Pool token embeddings to line-level via mean pooling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_embeddings, token_to_line, attention_mask, max_lines=None):
        """
        Args:
            token_embeddings: [B, T, H]
            token_to_line:    [B, T] line number per token (0 or -1 = ignore)
            attention_mask:   [B, T]
        Returns:
            line_embeddings:  [B, L, H]
            line_mask:        [B, L]
        """
        B, T, H = token_embeddings.shape
        device = token_embeddings.device

        if max_lines is None:
            max_lines = max(token_to_line.max().item(), 1) + 1

        line_emb = torch.zeros(B, max_lines, H, device=device)
        line_cnt = torch.zeros(B, max_lines, device=device)

        # Vectorized scatter-add
        for b in range(B):
            valid = (token_to_line[b] > 0) & (attention_mask[b] > 0)
            if not valid.any():
                continue
            indices = token_to_line[b][valid].long()  # [K]
            embeddings = token_embeddings[b][valid]    # [K, H]

            # Clamp to bounds
            indices = indices.clamp(0, max_lines - 1)
            line_emb[b].scatter_add_(0, indices.unsqueeze(1).expand(-1, H), embeddings)
            line_cnt[b].scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))

        line_cnt = line_cnt.clamp(min=1)
        line_emb = line_emb / line_cnt.unsqueeze(-1)
        line_mask = (line_cnt > 0).float()
        line_mask[:, 0] = 0  # Line 0 is padding/special

        line_emb = self.norm(line_emb)
        return line_emb, line_mask


class LightGATLayer(nn.Module):
    """Lightweight GAT layer for structural context (optional)."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, H = x.shape
        residual = x

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_2d = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(mask_2d == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, H)
        out = self.out(out)
        x = self.norm(residual + out)
        return x


class LLVT(nn.Module):
    """
    Line-Level Vulnerability Tagger.

    Pipeline:
    1. Token → Line Pooling (mean)
    2. Optional GAT context layers
    3. Per-line classification (sigmoid)
    4. Bag-level prediction (max-pool over line scores)
    5. Optional contrastive loss on line embeddings
    """

    def __init__(
        self,
        hidden_dim=768,
        num_gat_layers=1,
        num_heads=4,
        dropout=0.1,
        focal_gamma=2.0,
        focal_alpha=0.25,
        contrastive_weight=0.1,
        contrastive_temp=0.07,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.contrastive_weight = contrastive_weight

        # 1. Line Pooler
        self.line_pooler = LinePooler(hidden_dim)

        # 2. GAT context (lightweight)
        self.gat_layers = nn.ModuleList([
            LightGATLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gat_layers)
        ])

        # 3. Line classifier
        self.line_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # 4. Bag classifier (from max-pooled line features)
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 8),  # 8 vulnerability types
        )

        # Losses
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.contrastive_loss = SupConLoss(temperature=contrastive_temp)

        # Projection head for contrastive (projects to smaller space)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Total params
        total = sum(p.numel() for p in self.parameters())
        print(f"[LLVT] Initialized with {total:,} parameters")

    def forward(
        self,
        token_embeddings,
        token_to_line,
        attention_mask,
        line_labels=None,        # [B, L] binary per-line labels
        line_label_weights=None,  # [B, L] confidence per line (1.0 for curated, 0.3 for ScrawlD)
        bag_labels=None,          # [B, 8] vulnerability type labels
    ):
        """
        Returns dict with predictions and optionally losses.
        """
        # 1. Pool to line-level
        line_emb, line_mask = self.line_pooler(
            token_embeddings, token_to_line, attention_mask
        )
        B, L, H = line_emb.shape

        # 2. GAT context
        h = line_emb
        for gat in self.gat_layers:
            h = gat(h, line_mask)

        # 3. Per-line vulnerability score
        line_logits = self.line_classifier(h).squeeze(-1)  # [B, L]

        # 4. Bag-level prediction (max-pool over valid lines)
        masked_h = h * line_mask.unsqueeze(-1)
        # Use attention-weighted pool for bag
        bag_pool, _ = masked_h.max(dim=1)  # [B, H]
        bag_logits = self.bag_classifier(bag_pool)  # [B, 8]

        result = {
            "line_logits": line_logits,         # [B, L] raw logits
            "line_scores": torch.sigmoid(line_logits),  # [B, L] probabilities
            "bag_logits": bag_logits,            # [B, 8]
            "line_mask": line_mask,              # [B, L]
            "line_embeddings": h,                # [B, L, H] for visualization
        }

        # Compute losses if labels provided
        if line_labels is not None:
            # Focal loss per line
            focal = self.focal_loss(line_logits, line_labels.float())  # [B, L]
            focal = focal * line_mask  # ignore padding

            # Apply confidence weighting
            if line_label_weights is not None:
                focal = focal * line_label_weights

            # Negative sampling: randomly zero out 70% of easy-negative losses
            # Easy negatives = safe lines with low predicted probability
            with torch.no_grad():
                is_negative = (line_labels == 0) & (line_mask > 0)
                neg_mask = torch.rand_like(focal) > 0.7  # keep 30% of negatives
                keep_mask = (~is_negative) | neg_mask  # always keep positives
            focal = focal * keep_mask.float()

            n_valid = (line_mask * keep_mask.float()).sum().clamp(min=1)
            line_loss = focal.sum() / n_valid
            result["line_loss"] = line_loss

            # Contrastive loss on line embeddings
            if self.contrastive_weight > 0:
                contrastive_total = torch.tensor(0.0, device=h.device)
                for b in range(B):
                    proj = self.contrastive_proj(h[b])  # [L, 64]
                    cl = self.contrastive_loss(
                        proj, line_labels[b], line_mask[b]
                    )
                    contrastive_total = contrastive_total + cl
                contrastive_total = contrastive_total / B
                result["contrastive_loss"] = contrastive_total
            else:
                result["contrastive_loss"] = torch.tensor(0.0, device=h.device)

        if bag_labels is not None:
            bag_loss = F.binary_cross_entropy_with_logits(bag_logits, bag_labels)
            result["bag_loss"] = bag_loss

        return result
