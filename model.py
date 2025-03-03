import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings


# Add xformers import
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("successfully imported xformers")
except ImportError:
    XFORMERS_AVAILABLE = False


class TransferNetwork(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_classes=1):
        super(TransferNetwork, self).__init__()
        # obviously must match label shape
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        
        # Replace separate Conv2d layers with nn.Sequential for better performance
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, (5, 5), padding=2),
            nn.ReLU(inplace=True),  # inplace ReLU saves memory
            nn.LayerNorm([32, self.height, self.width]),
            nn.Conv2d(32, 16, (5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, (3, 3), padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer_norm_2 = torch.nn.LayerNorm(8 * self.height * self.width)
        self.classifier_out = torch.nn.Linear(8 * self.height * self.width, num_classes)

    def forward(self, embeddings):
        # Combine reshape and permute operations
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels).permute(0, 3, 1, 2)
        
        # Use sequential module
        embeddings = self.feature_extractor(embeddings)
        
        embeddings = embeddings.reshape(-1, 8 * self.height * self.width)
        embeddings = self.layer_norm_2(embeddings)
        return self.apply_last_layer(embeddings)

    def apply_last_layer(self, embeddings):
        return self.classifier_out(embeddings)


class Dinov2ForClassification(torch.nn.Module):
    def __init__(self, size, num_classes, classifier_type, cls_option="both"):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.dinov2_config = Dinov2Config.from_pretrained(f"facebook/dinov2-{size}")
        self.dinov2_config.attention_module = "xformers"
        self.dinov2 = Dinov2Model.from_pretrained(f"facebook/dinov2-{size}", config=self.dinov2_config)
        self.classifier = self.get_classifier()
        self.classifier_type = classifier_type
        if self.classifier_type == "bce":
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.dinov2(pixel_values)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]    # remove cls token
        # pass the output patch embeddings through the classifier
        logits = self.classifier(patch_embeddings)
        loss = None
        # don't upsample logits, because we instead downsample the labels (faster). If we are generating, upsample.
        if labels is not None:
            # don't squeeze batch dimension if it is of size 1, only squeeze the channel dimension
            logits = logits.squeeze(1)
            loss = self.loss_fct(logits, labels)

        return logits, loss

    def get_classifier(self):
        return TransferNetwork(self.dinov2_config.hidden_size, 18, 13, self.num_classes)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused_available = True  # wasn't working with dino so i disabled
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class TransferNetworkRNN(TransferNetwork):
    def __init__(self, in_channels, tokenW=32, tokenH=32):
        super().__init__(in_channels, tokenW, tokenH)
        self.feature_size = 8 * self.width * self.height
        del self.classifier_out     # so we can use all the weights from the non-lstm checkpoint easily

    def apply_last_layer(self, embeddings):
        return embeddings


class EfficientTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_xformers=True, max_seq_len=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = float(self.head_dim ** -0.5)
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        
        # Keep the same parameter initialization for both implementations
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        
        # RoPE (Rotary Position Embedding)
        self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(self, x, cls_attention_only=False):
        # Pre-norm for attention
        norm_x = self.norm1(x)
        batch_size, seq_len, hidden_size = norm_x.shape
        
        # Split the computation based on cls_attention_only for better efficiency
        if cls_attention_only:
            # Extract just what we need - CLS token only
            cls_token = x[:, 0:1]  # [batch, 1, hidden]
            
            # For CLS-only attention, only compute q for CLS token, but k,v for all tokens
            q_cls = self.q_proj(norm_x[:, 0:1])  # Only compute q for CLS token
            k = self.k_proj(norm_x)  # Compute k for all tokens
            v = self.v_proj(norm_x)  # Compute v for all tokens
            
            # Reshape for attention computation
            q_cls = q_cls.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, 1, head_dim]
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_len, head_dim]
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_len, head_dim]
            
            q_cls = self.rope(q_cls)
            k = self.rope(k)
            
            # Always use standard attention for CLS-only attention to avoid wasteful computation
            attention_scores = torch.matmul(q_cls, k.transpose(-1, -2)) * (1 / self.scale)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attn_output = torch.matmul(attention_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, hidden_size)
            
            # Project output and apply dropout (only for CLS token)
            attn_output = self.out_proj(attn_output)

            if self.use_dropout:
                attn_output = self.dropout(attn_output)
            
            # Residual connection (only for CLS token)
            cls_token = cls_token + attn_output
            # Apply norm and MLP (only for CLS token)
            mlp_out = self.mlp(self.norm2(cls_token))

            if self.use_dropout:
                mlp_out = self.dropout(mlp_out)

            # residual
            cls_token = cls_token + mlp_out
            return cls_token
        
        # Standard attention for all tokens (non-cls-only layers)
        else:
            # Compute Q, K, V for all tokens
            q = self.q_proj(norm_x)
            k = self.k_proj(norm_x)
            v = self.v_proj(norm_x)
            
            # Reshape for attention computation
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            q = self.rope(q)
            k = self.rope(k)
            
            if self.use_xformers:
                # xFormers memory-efficient attention
                attn_output = xops.memory_efficient_attention(
                    q, k, v, 
                    attn_bias=None,
                    scale=self.scale
                )
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            else:
                # Standard scaled dot-product attention
                attention_scores = torch.matmul(q, k.transpose(-1, -2)) * (1 / self.scale)
                attention_probs = F.softmax(attention_scores, dim=-1)
                attn_output = torch.matmul(attention_probs, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Common output projection
        attn_output = self.out_proj(attn_output)

        if self.use_dropout:
            attn_output = self.dropout(attn_output)

        # Residual connection and post-norm MLP
        x = x + attn_output
        mlp_out = self.mlp(self.norm2(x))

        if self.use_dropout:
            mlp_out = self.dropout(mlp_out)

        x = x + mlp_out
        return x

class EfficientTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0, use_xformers=True, max_seq_len=4096):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_size, num_heads, dropout, use_xformers, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.num_layers = num_layers
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Optimize the last layer by only computing CLS token attention
            cls_attention_only = (i == self.num_layers - 1)
            x = layer(x, cls_attention_only=cls_attention_only)
        return self.norm(x)

class Dinov2ForTimeSeriesClassification(Dinov2ForClassification):
    def __init__(self, size, num_classes, classifier_type, cls_option="patches_only", dropout_rate=0.1):
        super().__init__(size, num_classes, classifier_type)
        self.cls_option = cls_option
        
        # DinoV2 embedding dimension
        self.dino_embed_dim = self.dinov2_config.hidden_size  # 768 for DinoV2
        # Transformer parameters - keep fixed head count
        self.num_heads = 8
        self.num_layers = 6
        
        # Calculate exact context length based on patches and frames
        self.patches_per_frame = 13 * 18  # 234 patches per frame
        self.max_frames = 3
        
        # Determine transformer embedding dimension
        self.use_dino_embed_size = False
        self.transformer_dim = 128
        
        if self.use_dino_embed_size or self.transformer_dim is None:
            self.embed_dim = self.dino_embed_dim
        else:
            self.embed_dim = self.transformer_dim
            # Need dimension to be divisible by 2*num_heads for rotary embeddings, and some other limitations for xformers attention, but I decided against
            # auto fixing it, let the user set the correct transformer_dim!
            
        # Add projection layer from DinoV2 embedding to transformer embedding
        if self.dino_embed_dim != self.embed_dim:
            self.projection = nn.Linear(self.dino_embed_dim, self.embed_dim)
        else:
            self.projection = nn.Identity()
        
        # Set context length based on configuration
        if cls_option == "patches_only":
            # All patches + 1 CLS token
            self.context_length = self.patches_per_frame * self.max_frames + 1
        else:  # "both"
            # All patches + frame CLS tokens + 1 learnable CLS token
            self.context_length = self.patches_per_frame * self.max_frames + self.max_frames + 1
        
        # Learnable CLS token (uses transformer embedding dimension)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Transformer encoder with xformers
        self.transformer = EfficientTransformer(
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=dropout_rate,
            use_xformers=True,  # Ensure xformers is used
            max_seq_len=self.context_length
        )
        
        # Classification head (from transformer dimension to num_classes)
        self.fc_head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, labels=None):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process all frames through DinoV2
        x_reshaped = x.reshape(-1, channels, height, width)
        out = self.dinov2(x_reshaped)
        hidden_states = out.last_hidden_state
        
        # Project all hidden states at once
        projected = self.projection(hidden_states)
        
        # Reshape to separate frames
        projected = projected.reshape(batch_size, seq_len, -1, self.embed_dim)
        
        # Prepare learnable CLS token
        learnable_cls = self.cls_token.expand(batch_size, 1, -1)
        
        # Handle sequence composition based on configuration
        if self.cls_option == "patches_only":
            # Skip the DinoV2 CLS token (first token of each frame)
            patches = projected[:, :, 1:, :].reshape(batch_size, -1, self.embed_dim)
            sequence = torch.cat([learnable_cls, patches], dim=1)
        else:  # "both"
            # Keep all tokens (includes CLS tokens)
            all_tokens = projected.reshape(batch_size, -1, self.embed_dim)
            sequence = torch.cat([learnable_cls, all_tokens], dim=1)
        
        # Apply transformer
        cls_output = self.transformer(sequence)[:, 0]  # Shape: [batch, embed_dim] (squeezes size 1 dim in between, transformer output is already only "1" size due to only outputting cls token, but we have to squeeze the dim)
        
        # Final classification
        logits = self.fc_head(cls_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            last_label = labels[:, -1, :]
            loss = self.loss_fct(logits.view(-1, self.num_classes), last_label.view(-1, self.num_classes))
                
        return logits, loss

    def get_classifier(self):
        # No longer used in our implementation
        return None