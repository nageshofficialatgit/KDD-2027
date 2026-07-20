import torch
import torch.nn as nn
import torch.nn.functional as F

class VulnerabilityMILHead(nn.Module):
    """
    Multi-Instance Learning Head for Vulnerability Detection.
    
    Architecture:
    1. Instance Encoder: Projects each chunk's embedding (CLS or Pooler).
    2. Attention Layer: Computes scalar weight 'a_i' for each instance.
    3. Aggregation: Weighted sum of instance embeddings -> Bag Embedding.
    4. Classifier: Predicts vulnerability probability from Bag Embedding.
    
    Backprop flows through this head back to the GraphCodeBERT encoder.
    """
    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Instance Projector (Optional non-linearity before attention)
        self.instance_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 2. Attention Mechanism (Gated Attention is often better, using standard tanh for now)
        # w^T * tanh(V * h_i)
        self.attention_v = nn.Linear(hidden_dim, 128)
        self.attention_u = nn.Linear(128, 1) # scalar score
        
        # 3. Bag Classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Auxiliary Instance Classifier (Weak Supervision helper)
        # Can be used to force instances to be discriminative
        self.instance_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [Batch, Num_Instances, Hidden_Dim] - Embeddings of all chunks
            mask: [Batch, Num_Instances] - 1 for valid instances, 0 for padding
        """
        batch_size, num_instances, _ = x.shape
        
        # Project instances
        h = self.instance_projector(x) # [B, N, H]
        
        # Compute Attention Scores
        # A = softmax(u * tanh(v * h))
        attn_check = torch.tanh(self.attention_v(h)) # [B, N, 128]
        attn_scores = self.attention_u(attn_check).squeeze(-1) # [B, N]
        
        if mask is not None:
            # Mask padding instances
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1) # [B, N]
        
        # Aggregate
        # Weighted sum: Sum(a_i * h_i)
        # [B, N, 1] * [B, N, H] -> [B, N, H] -> sum -> [B, H]
        bag_embedding = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        
        # Bag Prediction
        bag_logits = self.bag_classifier(bag_embedding)
        
        # Instance Predictions (for auxiliary loss or visualization)
        # We generally predict logits for every instance
        instance_logits = self.instance_classifier(x) # [B, N, Num_Classes]
        
        return {
            "bag_logits": bag_logits,
            "bag_embedding": bag_embedding,
            "attention_weights": attn_weights, # Crucial for localization
            "instance_logits": instance_logits
        }

    def get_interpretable_attention(self, attn_weights, chunks_info):
        """
        Helper to map attention weights back to code lines (Human Readable).
        Args:
           attn_weights: [Num_Instances] tensor
           chunks_info: List of dicts with 'line_start', 'line_end'
        """
        results = []
        weights = attn_weights.detach().cpu().numpy()
        for i, weight in enumerate(weights):
            if i < len(chunks_info):
                info = chunks_info[i]
                results.append({
                    "chunk_id": i,
                    "score": float(weight),
                    "lines": (info.get("line_start"), info.get("line_end"))
                })
        return results
