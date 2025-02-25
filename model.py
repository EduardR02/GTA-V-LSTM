import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn as nn
import math
from torch.nn import functional as F


# Add xformers import
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
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
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_xformers=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        
        if self.use_xformers:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.dropout = dropout
        else:
            self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x):
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        
        if self.use_xformers:
            # Simple xformers implementation - always force CUDA
            batch_size, seq_len, _ = x.size()
            
            # Project to queries, keys, values
            q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Always force CUDA for xformers operations
            if not q.is_cuda:
                q = q.cuda()
                k = k.cuda()
                v = v.cuda()
            
            # Apply xformers memory-efficient attention
            attn_output = xops.memory_efficient_attention(
                q, k, v, 
                attn_bias=None, 
                p=self.dropout,
                scale=float(self.head_dim ** -0.5)
            )
                
            # Reshape output and project back to hidden_size
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
            x = self.out_proj(attn_output)
        else:
            # Standard PyTorch implementation
            x, _ = self.self_attn(x, x, x)
        
        x = residual + x
        
        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x

class EfficientTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0, use_xformers=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_size, num_heads, dropout, use_xformers)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Dinov2ForTimeSeriesClassification(Dinov2ForClassification):
    def __init__(self, size, num_classes, classifier_type, cls_option="both", dropout_rate=0.0, use_classifier=True):
        super().__init__(size, num_classes, classifier_type)
        self.cls_option = cls_option
        self.use_classifier = use_classifier
        
        # Determine input dimensions
        if not use_classifier and cls_option in ["patches_only", "both"]:
            # Use raw DinoV2 outputs (768 or 1024 dim)
            patch_dim = self.dinov2_config.hidden_size
        elif use_classifier and cls_option in ["patches_only", "both"]:
            # Use classifier reduced dimension
            patch_dim = self.classifier.feature_size
        else:
            patch_dim = 0
            
        cls_dim = self.dinov2_config.hidden_size if cls_option in ["cls_only", "both"] else 0
        
        # Total input dimension
        input_dim = patch_dim + cls_dim
        
        # Transformer configuration
        self.transformer_dim = 512
        self.num_heads = 8
        self.num_layers = 4
        
        # Projection layer (if needed)
        self.projection = nn.Linear(input_dim, self.transformer_dim) if input_dim != self.transformer_dim else nn.Identity()
        
        # Positional encoding
        self.register_buffer(
            "pos_embedding", 
            self.get_positional_embeddings(100, self.transformer_dim)
        )
        
        # Efficient transformer without dropout
        self.transformer = EfficientTransformer(
            hidden_size=self.transformer_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=dropout_rate
        )
        
        # Simplified classification head - just a single linear projection
        self.fc_head = nn.Linear(self.transformer_dim, num_classes)

    def get_positional_embeddings(self, seq_len, dim):
        """Return positional embeddings."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pos_embedding = torch.zeros(seq_len, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding
        
    def forward(self, x, labels=None):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process all frames at once through DinoV2
        x_reshaped = x.reshape(-1, channels, height, width)
        out = self.dinov2(x_reshaped)
        hidden_states = out.last_hidden_state
        
        # Choose features based on configuration
        features_list = []
        
        if self.cls_option in ["cls_only", "both"]:
            # Get CLS token features
            cls_features = hidden_states[:, 0, :].reshape(batch_size, seq_len, -1)
            features_list.append(cls_features)
        
        if self.cls_option in ["patches_only", "both"]:
            patches = hidden_states[:, 1:, :]
            if self.use_classifier:
                # Use classifier to reduce dimension
                processed_patches = self.classifier(patches)
                processed_patches = processed_patches.reshape(batch_size, seq_len, -1)
                features_list.append(processed_patches)
            else:
                # Use raw patch embeddings (mean pooling across patches)
                raw_patches = patches.mean(dim=1).reshape(batch_size, seq_len, -1)
                features_list.append(raw_patches)
        
        # Combine features
        combined_features = torch.cat(features_list, dim=-1) if len(features_list) > 1 else features_list[0]
        
        # Project to transformer dimension if needed
        features = self.projection(combined_features)
        
        # Add positional encoding
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)
        features = features + pos_emb
        
        # Apply transformer
        transformer_output = self.transformer(features)
        
        # Use last token representation for classification
        last_state = transformer_output[:, -1]
        
        # Classification
        logits = self.fc_head(last_state)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            last_label = labels[:, -1, :]
            loss = self.loss_fct(logits.view(-1, self.num_classes), last_label.view(-1, self.num_classes))
            
        return logits, loss

    def get_classifier(self):
        return TransferNetworkRNN(self.dinov2_config.hidden_size, 18, 13)
