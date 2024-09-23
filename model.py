import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel


class TransferNetwork(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_classes=1):
        super(TransferNetwork, self).__init__()
        # obviously must match label shape
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier_h1 = torch.nn.Conv2d(in_channels, 32, (5, 5), padding=2)
        self.classifier_h2 = torch.nn.Conv2d(32, 16, (5, 5), padding=2)
        self.classifier_h3 = torch.nn.Conv2d(16, 8, (3, 3), padding=1)
        self.classifier_out = torch.nn.Linear(8 * self.height * self.width, num_classes)

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = torch.nn.functional.relu(self.classifier_h1(embeddings))
        embeddings = torch.nn.functional.relu(self.classifier_h2(embeddings))
        embeddings = torch.nn.functional.relu(self.classifier_h3(embeddings))
        embeddings = embeddings.reshape(-1, 8 * self.height * self.width)
        return self.apply_last_layer(embeddings)

    def apply_last_layer(self, embeddings):
        return self.classifier_out(embeddings)


class Dinov2ForClassification(Dinov2PreTrainedModel):
    def __init__(self, config, classifier_type):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = TransferNetwork(config.hidden_size, 13, 18, config.num_labels)
        self.classifier_type = classifier_type
        if self.classifier_type == "bce":
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        # pass the output patch embeddings through the classifier
        logits = self.classifier(patch_embeddings)
        loss = None
        # don't upsample logits, because we instead downsample the labels (faster). If we are generating, upsample.
        if labels is not None:
            # don't squeeze batch dimension if it is of size 1, only squeeze the channel dimension
            logits = logits.squeeze(1)
            loss = self.loss_fct(logits, labels)

        return logits, loss

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


class TransferNetworkLSTM(TransferNetwork):
    def __init__(self, in_channels, tokenW=32, tokenH=32):
        super().__init__(in_channels, tokenW, tokenH)
        self.layer_norm = torch.nn.LayerNorm(8 * self.width * self.height)
        self.feature_size = 8 * self.width * self.height
        del self.classifier_out     # so we can use all the weights from the non-lstm checkpoint easily

    def apply_last_layer(self, embeddings):
        return self.layer_norm(embeddings)


class Dinov2ForTimeSeriesClassification(Dinov2ForClassification):
    def __init__(self, config, classifier_type):
        super().__init__(config, classifier_type)
        self.classifier = TransferNetworkLSTM(config.hidden_size, 13, 18)

        # LSTM layer
        self.lstm_hidden_size = 256
        self.lstm = torch.nn.LSTM(
            input_size=self.classifier.feature_size,  # Features + previous labels
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False
        )

        # Final classification layer
        self.final_linear_layer = torch.nn.Linear(self.lstm_hidden_size, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        batch_size, time_steps, channels, height, width = pixel_values.shape

        # Process each time step
        features = []
        for t in range(time_steps):
            outputs = self.dinov2(
                pixel_values[:, t],
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]
            features.append(self.classifier(patch_embeddings))

        # Stack features from all time steps
        features = torch.stack(features, dim=1)  # Shape: (batch_size, time_steps, feature_size)
        # Prepare input for LSTM (including previous labels if available)
        if labels is not None:
            # Assume labels are of shape (batch_size, time_steps, num_labels)
            # all_except_last_label = labels[:, :-1, :]
            # lstm_input = torch.cat([features, labels], dim=-1)
            pass
        else:
            # If no labels available (e.g., during inference), use zeros
            # dummy_labels = torch.zeros(batch_size, time_steps, self.config.num_labels, device=features.device)
            # lstm_input = torch.cat([features, dummy_labels], dim=-1)
            pass
        # Process through LSTM
        lstm_out, _ = self.lstm(features)
        lstm_last_output = lstm_out[:, -1, :]

        # Final classification
        logits = self.final_linear_layer(lstm_last_output)

        loss = None
        if labels is not None:
            last_label = labels[:, -1, :]
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), last_label.view(-1, self.config.num_labels))

        return logits, loss
