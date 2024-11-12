import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadSelfAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                           attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class HKLEmbedding(nn.Module):
    def __init__(self, input_shape, embed_dim, embedding_type='learned'):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        
        # Generate coordinate meshgrid
        h = torch.arange(input_shape[0])
        k = torch.arange(input_shape[1])
        l = torch.arange(input_shape[2])
        
        # Create meshgrid of indices
        self.h, self.k, self.l = torch.meshgrid(h, k, l, indexing='ij')
        
        if embedding_type == 'onehot':
            # Create fixed one-hot encodings
            self.register_buffer('h_onehot', torch.eye(input_shape[0]))
            self.register_buffer('k_onehot', torch.eye(input_shape[1]))
            self.register_buffer('l_onehot', torch.eye(input_shape[2]))
            
            # Simple linear projections to desired dimension
            self.h_proj = nn.Linear(input_shape[0], embed_dim // 3)
            self.k_proj = nn.Linear(input_shape[1], embed_dim // 3)
            self.l_proj = nn.Linear(input_shape[2], embed_dim - 2*(embed_dim // 3))
        else:  # learned
            self.hkl_proj = nn.Sequential(
                nn.Linear(3, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
    
    def forward(self, x):
        batch_size = x.shape[0]
        H, W, D = self.input_shape

        if self.embedding_type == 'onehot':
            # Get one-hot vectors
            h_emb = self.h_onehot[self.h]  # [H, W, D, H]
            k_emb = self.k_onehot[self.k]  # [H, W, D, W]
            l_emb = self.l_onehot[self.l]  # [H, W, D, D]

            # Project to embedding dimension
            h_emb = self.h_proj(h_emb)  # [H, W, D, embed_dim//3]
            k_emb = self.k_proj(k_emb)  # [H, W, D, embed_dim//3]
            l_emb = self.l_proj(l_emb)  # [H, W, D, embed_dim//3]

            # Concatenate along embedding dimension
            hkl_embedding = torch.cat([h_emb, k_emb, l_emb], dim=-1)  # [H, W, D, embed_dim]

            # Reshape to match value embedding shape
            hkl_embedding = hkl_embedding.view(-1, self.embed_dim)  # [H*W*D, embed_dim]

            # Add batch dimension and expand
            hkl_embedding = hkl_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W*D, embed_dim]

            return hkl_embedding
        else:
            # Learned embedding approach
            hkl_coords = torch.stack([self.h, self.k, self.l], dim=-1).float()  # [H, W, D, 3]
            hkl_embedding = self.hkl_proj(hkl_coords.reshape(-1, 3))  # [H*W*D, embed_dim]

            # Add batch dimension and expand
            hkl_embedding = hkl_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W*D, embed_dim]

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
                 embedding_type='onehot'):  # 'learned' or 'onehot'
        super().__init__()
        
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.gradient_checkpointing = True
        
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
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        x = self.norm(x)
        
        # Project to output space
        x = self.output_proj(x)  # [B, H*W*D, 1]
        
        # Reshape back to 3D
        x = x.reshape(B, H, W, D, 1).permute(0, 4, 1, 2, 3)
        
        # Apply mask to preserve input values
        # Return original values where mask is 1, and predicted values where mask is 0
        return torch.where(mask == 1, input_x, x)