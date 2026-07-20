#!/usr/bin/env python3
"""
Graph-Aware Multi-Instance Learning Head for Vulnerability Localization.

Architecture:
1. Token → Line Pooling: Pool token embeddings to line-level representations
2. Line-Level Graph Construction: Build adjacency from graph features (WIRE coords)
3. GAT Message Passing: K rounds of Graph Attention over line representations  
4. Gated Multi-Head Attention: Compute vulnerability scores per line
5. Bag Aggregation: Weighted sum for bag-level classification

This replaces the shallow VulnerabilityMILHead with graph-aware reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinePooler(nn.Module):
    """Pool token embeddings to line-level representations using line_sink_mask."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, token_embeddings, token_to_line, attention_mask):
        """
        Args:
            token_embeddings: [B, T, H]
            token_to_line: [B, T] - line index per token (-1 for special/pad)
            attention_mask: [B, T]
        Returns:
            line_embeddings: [B, max_lines, H]
            line_mask: [B, max_lines] - 1 for valid lines, 0 for padding
            line_indices: list of unique line indices per batch element
        """
        B, T, H = token_embeddings.shape
        device = token_embeddings.device
        
        # Find max line index across batch
        max_line = token_to_line.max().item()
        if max_line < 1:
            max_line = 1
        num_lines = max_line + 1  # lines are 1-indexed, but we include 0 for safety
        
        # Aggregate tokens per line via mean pooling
        line_embeddings = torch.zeros(B, num_lines, H, device=device)
        line_counts = torch.zeros(B, num_lines, device=device)
        
        for b in range(B):
            valid = (token_to_line[b] > 0) & (attention_mask[b] > 0)
            valid_indices = valid.nonzero(as_tuple=True)[0]
            
            for idx in valid_indices:
                line_idx = token_to_line[b, idx].item()
                if 0 < line_idx < num_lines:
                    line_embeddings[b, line_idx] += token_embeddings[b, idx]
                    line_counts[b, line_idx] += 1
        
        # Avoid division by zero
        line_counts = line_counts.clamp(min=1)
        line_embeddings = line_embeddings / line_counts.unsqueeze(-1)
        
        # Line mask: lines that have at least one token
        line_mask = (line_counts > 0).float()
        # Zero out line 0 (it's typically special tokens)
        line_mask[:, 0] = 0
        
        line_embeddings = self.norm(self.proj(line_embeddings))
        
        return line_embeddings, line_mask


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer for line-level message passing.
    Uses WIRE graph features to determine edge weights.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection (for incorporating structural distance)
        self.edge_proj = nn.Linear(1, num_heads, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, adj_matrix, line_mask):
        """
        Args:
            x: [B, L, H] line embeddings
            adj_matrix: [B, L, L] adjacency/similarity matrix
            line_mask: [B, L] 
        Returns:
            x: [B, L, H] updated embeddings
        """
        B, L, H = x.shape
        residual = x
        
        # Multi-head attention with graph bias
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add graph structure bias from adjacency
        # adj_matrix: [B, L, L] → edge bias: [B, num_heads, L, L]
        graph_bias = self.edge_proj(adj_matrix.unsqueeze(-1))  # [B, L, L, num_heads]
        graph_bias = graph_bias.permute(0, 3, 1, 2)  # [B, num_heads, L, L]
        attn = attn + graph_bias
        
        # Mask invalid lines
        mask_2d = line_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn = attn.masked_fill(mask_2d == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # [B, num_heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.W_o(out)
        
        # Residual + Norm
        x = self.norm(residual + out)
        
        # FFN
        x = self.ffn_norm(x + self.ffn(x))
        
        return x


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Mechanism (Ilse et al., 2018).
    attention = softmax(w^T * (tanh(V*h) ⊙ sigmoid(U*h)))
    
    Superior to standard attention for MIL localization.
    """
    
    def __init__(self, hidden_dim: int, attn_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        
        # Per-head gated attention
        self.attention_V = nn.ModuleList([
            nn.Linear(hidden_dim, attn_dim) for _ in range(num_heads)
        ])
        self.attention_U = nn.ModuleList([
            nn.Linear(hidden_dim, attn_dim) for _ in range(num_heads)
        ])
        self.attention_w = nn.ModuleList([
            nn.Linear(attn_dim, 1) for _ in range(num_heads)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, H] 
            mask: [B, L] - 1 for valid, 0 for padding
        Returns:
            attention_weights: [B, num_heads, L]
            bag_embeddings: [B, num_heads, H]
        """
        all_weights = []
        all_bags = []
        
        for head_idx in range(self.num_heads):
            h_V = torch.tanh(self.attention_V[head_idx](x))  # [B, L, attn_dim]
            h_U = torch.sigmoid(self.attention_U[head_idx](x))  # [B, L, attn_dim]
            
            # Gated attention: tanh(V*h) ⊙ sigmoid(U*h)
            scores = self.attention_w[head_idx](h_V * h_U).squeeze(-1)  # [B, L]
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            weights = F.softmax(scores, dim=1)  # [B, L]
            
            # Weighted sum
            bag = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # [B, H]
            
            all_weights.append(weights)
            all_bags.append(bag)
        
        attention_weights = torch.stack(all_weights, dim=1)  # [B, num_heads, L]
        bag_embeddings = torch.stack(all_bags, dim=1)  # [B, num_heads, H]
        
        return attention_weights, bag_embeddings


class GraphMILHead(nn.Module):
    """
    Graph-Aware Multi-Instance Learning Head.
    
    Pipeline:
    1. Pool tokens to line-level representations
    2. Build structural adjacency from graph features
    3. Run GAT message passing (K rounds)
    4. Gated multi-head attention for localization scores
    5. Bag-level classification from aggregated embedding
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 8,
        num_gat_layers: int = 2,
        num_attn_heads: int = 4,
        graph_feature_dim: int = 6,
        adjacency_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.adjacency_threshold = adjacency_threshold
        
        # 1. Token → Line Pooler
        self.line_pooler = LinePooler(hidden_dim)
        
        # 2. Graph feature projection for adjacency construction
        self.graph_feat_proj = nn.Sequential(
            nn.Linear(graph_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # 3. GAT Message Passing Layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads=num_attn_heads, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        
        # 4. Gated Attention Pooling
        self.gated_attention = GatedAttentionPooling(
            hidden_dim, attn_dim=128, num_heads=num_attn_heads
        )
        
        # 5. Bag Classifier (from multi-head bag embeddings)
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_attn_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # 6. Instance (Line) Classifier for auxiliary supervision
        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def _build_adjacency(self, line_graph_features, line_mask):
        """
        Build line-level adjacency matrix from average graph features per line.
        Lines with similar WIRE coordinates are structurally related.
        
        Args:
            line_graph_features: [B, L, 6]
            line_mask: [B, L]
        Returns:
            adj_matrix: [B, L, L] similarity-based adjacency
        """
        # Project graph features to a similarity space
        proj = self.graph_feat_proj(line_graph_features)  # [B, L, 32]
        
        # Cosine similarity between all line pairs
        proj_norm = F.normalize(proj, p=2, dim=-1)
        sim_matrix = torch.bmm(proj_norm, proj_norm.transpose(1, 2))  # [B, L, L]
        
        # Apply threshold for sparsity (soft adjacency)
        # Also add self-loops
        identity = torch.eye(sim_matrix.size(1), device=sim_matrix.device).unsqueeze(0)
        adj_matrix = torch.clamp(sim_matrix, min=0) + identity
        
        # Mask invalid lines
        mask_2d = line_mask.unsqueeze(1) * line_mask.unsqueeze(2)  # [B, L, L]
        adj_matrix = adj_matrix * mask_2d
        
        return adj_matrix
    
    def _pool_graph_features_to_lines(self, graph_features, token_to_line, attention_mask, num_lines):
        """Pool per-token graph features to line-level averages."""
        B, T, F = graph_features.shape
        device = graph_features.device
        
        line_graph_feats = torch.zeros(B, num_lines, F, device=device)
        line_counts = torch.zeros(B, num_lines, device=device)
        
        for b in range(B):
            valid = (token_to_line[b] > 0) & (attention_mask[b] > 0)
            valid_indices = valid.nonzero(as_tuple=True)[0]
            
            for idx in valid_indices:
                line_idx = token_to_line[b, idx].item()
                if 0 < line_idx < num_lines:
                    line_graph_feats[b, line_idx] += graph_features[b, idx]
                    line_counts[b, line_idx] += 1
        
        line_counts = line_counts.clamp(min=1)
        line_graph_feats = line_graph_feats / line_counts.unsqueeze(-1)
        
        return line_graph_feats
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_to_line: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_features: torch.Tensor = None,
    ):
        """
        Args:
            token_embeddings: [B, T, H] from encoder
            token_to_line: [B, T] line index per token
            attention_mask: [B, T]
            graph_features: [B, T, 6] WIRE structural features (optional)
        Returns:
            dict with bag_logits, attention_weights, instance_logits, line_mask
        """
        # 1. Pool to line-level
        line_embeddings, line_mask = self.line_pooler(
            token_embeddings, token_to_line, attention_mask
        )
        # line_embeddings: [B, L, H], line_mask: [B, L]
        
        B, L, H = line_embeddings.shape
        
        # 2. Build structural adjacency
        if graph_features is not None:
            line_graph_feats = self._pool_graph_features_to_lines(
                graph_features, token_to_line, attention_mask, L
            )
            adj_matrix = self._build_adjacency(line_graph_feats, line_mask)
        else:
            # Fallback: fully connected (like standard attention)
            adj_matrix = (line_mask.unsqueeze(1) * line_mask.unsqueeze(2))
        
        # 3. GAT Message Passing
        h = line_embeddings
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj_matrix, line_mask)
        
        # 4. Gated Attention for Localization
        attn_weights, bag_embeddings = self.gated_attention(h, line_mask)
        # attn_weights: [B, num_heads, L]
        # bag_embeddings: [B, num_heads, H]
        
        # 5. Bag Classification
        bag_concat = bag_embeddings.view(B, -1)  # [B, num_heads * H]
        bag_logits = self.bag_classifier(bag_concat)  # [B, num_classes]
        
        # 6. Instance (Line) Classification
        instance_logits = self.instance_classifier(h)  # [B, L, num_classes]
        
        # Average attention across heads for localization
        avg_attn = attn_weights.mean(dim=1)  # [B, L]
        
        return {
            "bag_logits": bag_logits,
            "attention_weights": avg_attn,
            "per_head_attention": attn_weights,
            "instance_logits": instance_logits,
            "line_embeddings": h,
            "line_mask": line_mask,
            "bag_embedding": bag_concat,
        }
