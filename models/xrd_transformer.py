import torch
import torch.nn as nn
import torch.nn.functional as F

class HKLEmbedding(nn.Module):
    def __init__(self, input_shape, embed_dim, embedding_type='learned'):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        
        # Generate normalized coordinate meshgrid
        h = torch.arange(input_shape[0])
        k = torch.arange(input_shape[1])
        l = torch.arange(input_shape[2])
        
        # Create meshgrid of indices
        self.h, self.k, self.l = torch.meshgrid(h, k, l, indexing='ij')
        
        if embedding_type == 'onehot':
            # One-hot encoding approach
            self.h_embed = nn.Embedding(input_shape[0], embed_dim // 3)
            self.k_embed = nn.Embedding(input_shape[1], embed_dim // 3)
            self.l_embed = nn.Embedding(input_shape[2], embed_dim // 3)
        else:  # learned
            # Learned embedding approach
            self.hkl_proj = nn.Sequential(
                nn.Linear(3, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Move indices to device
        h = self.h.to(x.device)
        k = self.k.to(x.device)
        l = self.l.to(x.device)
        
        if self.embedding_type == 'onehot':
            # One-hot embedding approach
            h_emb = self.h_embed(h.flatten())
            k_emb = self.k_embed(k.flatten())
            l_emb = self.l_embed(l.flatten())
            
            # Concatenate embeddings
            hkl_embedding = torch.cat([h_emb, k_emb, l_emb], dim=-1)
        else:
            # Learned embedding approach
            hkl_coords = torch.stack([h, k, l], dim=-1).float()  # Shape: [H, W, D, 3]
            hkl_embedding = self.hkl_proj(hkl_coords.reshape(-1, 3))
        
        # Expand for batch dimension
        hkl_embedding = hkl_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return hkl_embedding

class XRDTransformer(nn.Module):
    def __init__(self, 
                 input_shape=(26, 18, 23),
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 embedding_type='learned'):  # 'learned' or 'onehot'
        super().__init__()
        
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        
        # HKL Embedding
        self.hkl_embedding = HKLEmbedding(input_shape, embed_dim, embedding_type)
        
        # Value embedding
        self.value_proj = nn.Linear(1, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, True, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, 1)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Store input mask for preserving center values
        mask = (x != 0).float()
        input_x = x.clone()
        
        # Reshape input to sequence
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, 1)
        
        # Get HKL embeddings
        hkl_emb = self.hkl_embedding(x)  # [B, H*W*D, embed_dim]
        
        # Project values
        val_emb = self.value_proj(x)  # [B, H*W*D, embed_dim]
        
        # Combine embeddings
        x = hkl_emb + val_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Project to output space
        x = self.output_proj(x)  # [B, H*W*D, 1]
        
        # Reshape back to 3D
        x = x.reshape(B, H, W, D, 1).permute(0, 4, 1, 2, 3)
        
        # Apply mask to preserve input values
        # Return original values where mask is 1, and predicted values where mask is 0
        return torch.where(mask == 1, input_x, x)