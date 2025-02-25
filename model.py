import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn as nn


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


class Dinov2ForTimeSeriesClassification(Dinov2ForClassification):
    def __init__(self, size, num_classes, classifier_type, cls_option="both"):
        """
        After lots of confusion, pytorch stateless lstm is the default, and h_0 and c_0 are initialized to 0
        at the start of each SEQUENCE. There seems to be a lot of confusion on the internet likely from
        people calling sequences "batches" (which is a very big difference in this case).
        I was confused because if it was batches, you would have to
        preserve sample order in a batch and sample sequentially inside of a batch. But pytorch docs (for me) clears
        this up, because the h_0 and c_0 tensors are (D∗num_layers,N,Hout). Because we have N as a dim here that means
        that it's zero for each sequence.
        My initial investigation came from the fact that the loss function would dip (good, but weird) after each time
        the dataset would "end" and the next iter(dataloader) would be called. I thought this would cause some
        lstm hidden state contamination, but this doesn't seem to be the issue as the lstm is stateless here.

        Update: I think the loss issue comes from the dataset sampler. Apparently I thought that by default
        replacement should be True, which means that a sample can be drawn multiple times
        (and the len param of the dataset is "artificial"). This is not the case.
        By default it is False, so each dataloader actually goes through the entire dataset once.
        The loss jump is just normal epoch behavior then.
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
        super().__init__(size, num_classes, classifier_type)
        self.cls_option = cls_option
        self.rnn_hidden_size = 384
        # self.dropout = torch.nn.Dropout(0.2)
        if cls_option == "cls_only":
            del self.classifier
            input_size = self.dinov2_config.hidden_size
        elif cls_option == "patches_only":
            input_size = self.classifier.feature_size
        elif cls_option == "both":
            input_size = self.dinov2_config.hidden_size + self.classifier.feature_size
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False
        )
        
        # Combine FC layers into a sequential module
        self.fc_head = nn.Sequential(
            nn.LayerNorm(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, self.num_classes)
        )

    def forward(self, x, labels=None):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process all frames at once
        x_reshaped = x.reshape(-1, channels, height, width)
        out = self.dinov2(x_reshaped)
        hidden_states = out.last_hidden_state
        
        features = None
        
        # More efficient feature extraction
        if self.cls_option in ["patches_only", "both"]:
            processed_patches = self.classifier(hidden_states[:, 1:, :])
            processed_patches = processed_patches.reshape(batch_size, seq_len, -1)
            
        if self.cls_option in ["cls_only", "both"]:
            cls_features = hidden_states[:, 0, :].reshape(batch_size, seq_len, -1)
            
        # Set features based on cls_option
        if self.cls_option == "both":
            features = torch.cat([cls_features, processed_patches], dim=-1)
        elif self.cls_option == "cls_only":
            features = cls_features
        else:  # patches_only
            features = processed_patches
        
        # Process through RNN and get final timestep output
        outputs, _ = self.rnn(features)
        
        # Use sequential for the final processing
        logits = self.fc_head(outputs[:, -1, :])

        loss = None
        if labels is not None:
            last_label = labels[:, -1, :]
            loss = self.loss_fct(logits.view(-1, self.num_classes), last_label.view(-1, self.num_classes))

        return logits, loss

    def get_classifier(self):
        return TransferNetworkRNN(self.dinov2_config.hidden_size, 18, 13)
