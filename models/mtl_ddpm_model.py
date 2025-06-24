import torch
import os
import torch.nn.functional as F
from torch import nn
from .char_lstm import CharLSTM
from .bert_model import HMNeTNERModel
from .unimo_model import UnimoCRFModel
from utils.attention import MultiAttn, PositionalEncoding

# Class to manage the noise schedule for the diffusion process
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu", schedule_type="cosine"):
        """Initialize noise scheduler with cosine or linear schedule.
        
        Args:
            timesteps (int): Number of diffusion steps.
            beta_start (float): Starting noise variance.
            beta_end (float): Ending noise variance.
            device (str): Device to run computations (e.g., 'cpu', 'cuda').
            schedule_type (str): Type of noise schedule ('cosine' or 'linear').
        """
        self.timesteps = timesteps
        self.device = device
        assert 0 < beta_start < beta_end < 1, "Invalid beta range"
        
        # Create noise schedule based on type
        if schedule_type == "cosine":
            t = torch.linspace(0, 1, timesteps + 1, device=device)[:-1]
            self.beta = 1 - torch.cos(t * torch.pi / 2) ** 2
            self.beta = self.beta * (beta_end - beta_start) + beta_start
        else:
            self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
            
        # Compute alpha and cumulative alpha for diffusion
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.noise_scale = 1.0

    def set_noise_scale(self, scale):
        """Set the scale for noise addition.
        
        Args:
            scale (float): Scaling factor for noise magnitude.
        """
        self.noise_scale = scale

    def add_noise(self, x, t, attention_mask=None):
        """Add noise to input embeddings at timestep t, respecting attention mask.
        
        Args:
            x (torch.Tensor): Input embeddings to corrupt.
            t (torch.Tensor): Timestep indices.
            attention_mask (torch.Tensor, optional): Mask for valid tokens (1 for valid, 0 for padding).
        
        Returns:
            tuple: Noisy embeddings and added noise.
        """
        assert t.max() < self.timesteps and t.min() >= 0, "Invalid timestep"
        noise = torch.randn_like(x) * self.noise_scale
        signal_rate_t = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        noise_rate_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        noisy_x = signal_rate_t * x + noise_rate_t * noise
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            noisy_x = mask * noisy_x + (1 - mask) * x
        return noisy_x, noise

# Diffusion model for error detection in NER labels
class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_embedding_table=None, clstm_path=None, ner_model_name="hvpnet"):
        """Initialize the diffusion model with label, character, and visual-textual encoders.
        
        Args:
            args: Training arguments (e.g., device, hidden dimensions).
            num_labels (int): Number of NER labels.
            label_embedding_table (torch.Tensor): Pretrained label embeddings.
            clstm_path (str): Path to character LSTM model.
            ner_model_name (str): NER model type ('hvpnet' or 'mkgformer').
        """
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.ner_model_name = ner_model_name
        
        # Time embedding MLP for diffusion steps
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            timesteps=self.args.train_steps, 
            device=self.args.device,
            schedule_type="cosine"
        )
        
        # Label encoder: converts label indices to embeddings
        self.label_embedding_table = label_embedding_table
        self.label_mlp = nn.Sequential(
            nn.Linear(768, self.args.label_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.label_hidden_dim, self.args.label_hidden_dim)
        )
        self.label_pos_encoder = PositionalEncoding(self.args.label_hidden_dim, self.args.max_seq_len)
        self.label_self_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim, 
            key_dim=self.args.label_hidden_dim, 
            value_dim=self.args.label_hidden_dim, 
            emb_dim=self.args.label_hidden_dim, 
            num_heads=1, 
            dropout_rate=0.3
        )
        
        # Character LSTM encoder: processes character-level input
        char2int_dict, int2char_dict = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        self.char_lstm = CharLSTM(char2int_dict, int2char_dict, n_hidden=args.char_hidden_dim, 
                                 n_layers=2, bidirectional=True, drop_prob=0.3)
        self.char_lstm.load_state_dict(torch.load(os.path.join(clstm_path, "char_lstm.pth")))
        self.char_lstm_mlp = nn.Linear(2 * self.char_lstm.n_layers * args.char_hidden_dim, args.char_hidden_dim)
        self.char_pos_encoder = PositionalEncoding(args.char_hidden_dim, self.args.max_seq_len)
        self.char_self_attn = MultiAttn(
            query_dim=args.char_hidden_dim, 
            key_dim=args.char_hidden_dim, 
            value_dim=args.char_hidden_dim, 
            emb_dim=args.char_hidden_dim, 
            num_heads=4, 
            dropout_rate=0.3
        )
        
        # Visual-textual encoder: processes text and image inputs
        if ner_model_name == "hvpnet":
            self.ner_model = HMNeTNERModel(num_labels, args)
            self.vt_encoder = self.ner_model.core
            vt_hidden_size = self.vt_encoder.bert.config.hidden_size
        elif ner_model_name == "mkgformer":
            self.ner_model = UnimoCRFModel(num_labels, args)
            self.vt_encoder = self.ner_model.model
            vt_hidden_size = self.vt_encoder.text_config.hidden_size
        else:
            raise ValueError("Invalid ner_model_name")
        self.vt_hidden_size = vt_hidden_size
        
        # Cross-attention layers for integrating features
        self.label_vt_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,
            key_dim=vt_hidden_size,
            value_dim=vt_hidden_size,
            emb_dim=self.args.label_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.label_char_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,
            key_dim=self.args.char_hidden_dim,
            value_dim=self.args.char_hidden_dim,
            emb_dim=self.args.label_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.label_context_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=self.args.label_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.vt_label_attn = MultiAttn(
            query_dim=vt_hidden_size,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=vt_hidden_size,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.vt_char_attn = MultiAttn(
            query_dim=vt_hidden_size,
            key_dim=self.args.char_hidden_dim,
            value_dim=self.args.char_hidden_dim,
            emb_dim=vt_hidden_size,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.vt_context_attn = MultiAttn(
            query_dim=vt_hidden_size,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=vt_hidden_size,
            num_heads=4, 
            dropout_rate=0.4
        )
        
        # Normalization layers for attention outputs
        self.norm_label_vt = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_char = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_context = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_vt_label = nn.LayerNorm(self.vt_hidden_size)
        self.norm_vt_char = nn.LayerNorm(self.vt_hidden_size)
        self.norm_vt_context = nn.LayerNorm(self.vt_hidden_size)
        
        # Output layers for predictions
        self.fc = nn.Sequential(
            nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, num_labels),
            nn.LayerNorm(num_labels)
        )
        self.noise_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, self.args.label_hidden_dim)
        self.error_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.label_smoothing = 0.1
        
        # Valid labels for noise injection (exclude special tokens)
        self.valid_labels = torch.tensor([i for i in range(num_labels) if i not in [0, 10, 11, 12]], device=args.device)

    def get_label_embedding(self, labels, attention_mask=None):
        """Convert label indices to embeddings with positional encoding and self-attention.
        
        Args:
            labels (torch.Tensor): NER label indices.
            attention_mask (torch.Tensor, optional): Mask for valid tokens.
        
        Returns:
            torch.Tensor: Label embeddings.
        """
        assert labels is not None, "labels required"
        assert labels.max() < self.label_embedding_table.shape[0], "Label indices out of range"
        label_features = self.label_embedding_table[labels]
        label_features = self.label_mlp(label_features)
        label_features = self.label_pos_encoder(label_features)
        label_mask = (~attention_mask.bool()) if attention_mask is not None else None
        label_features = self.label_self_attn(
            query=label_features, key=label_features, value=label_features, mask=label_mask
        )
        return label_features

    def get_context_embedding(self, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, 
                             images=None, aux_imgs=None, rcnn_imgs=None, context_labels=None):
        """Generate embeddings for characters, visual-textual input, and context labels.
        
        Args:
            char_input_ids (torch.Tensor, optional): Character-level input IDs.
            input_ids (torch.Tensor): Token input IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            images (torch.Tensor, optional): Main images.
            aux_imgs (torch.Tensor, optional): Auxiliary images.
            rcnn_imgs (torch.Tensor, optional): RCNN images.
            context_labels (torch.Tensor, optional): Noisy NER labels used as context.
        
        Returns:
            tuple: Character, visual-textual, and context embeddings.
        """
        assert input_ids is not None, "input_ids required"
        bsz = input_ids.size(0)
        
        # Process character inputs
        if char_input_ids is not None:
            vocab_size = len(self.char_lstm.char2int)
            assert char_input_ids.shape[1:] == (self.args.max_seq_len, self.args.max_char_len), "Char input shape mismatch"
            char_input = F.one_hot(char_input_ids, vocab_size).view(-1, self.args.max_char_len, vocab_size).to(torch.float32)
            hc = self.char_lstm.init_hidden((char_input.shape[0],))
            hc = tuple([each.to(self.args.device) for each in hc])
            _, char_hidden = self.char_lstm(char_input, hc)
            char_features = char_hidden[0].transpose(0, 2).contiguous().view(bsz, self.args.max_seq_len, -1)
            char_features = self.char_lstm_mlp(char_features)
            char_features = self.char_pos_encoder(char_features)
            attn_mask = (~attention_mask.bool()) if attention_mask is not None else None
            char_features = self.char_self_attn(char_features, char_features, char_features, mask=attn_mask)
        else:
            char_features = None
        
        # Process visual-textual inputs
        if self.ner_model_name == "hvpnet":
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                  pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
            vt_features = out.last_hidden_state
        assert vt_features.shape == (bsz, self.args.max_seq_len, self.vt_hidden_size), "VT output shape mismatch"
        
        # Process context labels (noisy labels with errors)
        context_features = None
        if context_labels is not None:
            context_features = self.get_label_embedding(context_labels, attention_mask)
        
        return char_features, vt_features, context_features

    def compute_contrastive_loss(self, features, error_true_mask, attention_mask):
        """Compute contrastive loss to distinguish erroneous from correct tokens.
        
        Args:
            features (torch.Tensor): Combined features from denoise.
            error_true_mask (torch.Tensor): Ground truth error mask.
            attention_mask (torch.Tensor): Mask for valid tokens.
        
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        dim = features.shape[-1]
        features = features.view(-1, dim)
        error_true_mask = error_true_mask.view(-1)
        mask = attention_mask.view(-1).bool()
        
        features = features[mask]
        error_true_mask = error_true_mask[mask]
        
        features = F.normalize(features, dim=-1)
        sim_matrix = torch.matmul(features, features.t())
        label_matrix = (error_true_mask.unsqueeze(0) == error_true_mask.unsqueeze(1)).float()
        temperature = 0.07
        sim_matrix = sim_matrix / temperature
        pos_sim = sim_matrix * label_matrix
        neg_sim = sim_matrix * (1 - label_matrix)
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        loss = -torch.log(pos_exp.sum(dim=-1) / (pos_exp.sum(dim=-1) + neg_exp.sum(dim=-1) + 1e-8))
        return loss.mean()

    def corrupt(self, t, labels, targets_noise, attention_mask, mode='pretrain'):
        """Corrupt labels with diffusion noise; use targets_noise for context.
        
        Args:
            t (torch.Tensor): Timestep indices.
            labels (torch.Tensor): Clean NER labels (targets_unk for pretrain, targets_new for finetune).
            targets_noise (torch.Tensor): Corrupted NER labels (targets_noise for pretrain, targets_unk for finetune).
            attention_mask (torch.Tensor): Mask for valid tokens.
            mode (str): 'pretrain' or 'finetune'.
        
        Returns:
            tuple: Corrupted embeddings, noise, error mask.
        """
        label_features = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        
        # Compute error mask: difference between clean and noisy labels
        error_true_mask = (labels != targets_noise).float()
        return corrupt_label_embeddings, noise, error_true_mask

    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, context_labels=None, return_features=False):
        """Denoise corrupted embeddings, predicting labels, noise, and error mask.
        
        Args:
            corrupt_label_embeddings (torch.Tensor): Noisy label embeddings.
            t (torch.Tensor): Timestep indices.
            char_input_ids (torch.Tensor, optional): Character input IDs.
            input_ids (torch.Tensor): Token input IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            images (torch.Tensor, optional): Main images.
            aux_imgs (torch.Tensor, optional): Auxiliary images.
            rcnn_imgs (torch.Tensor, optional): RCNN images.
            context_labels (torch.Tensor, optional): Noisy NER labels used as context to guide denoising.
            return_features (bool): Whether to return combined features.
        
        Returns:
            tuple: NER logits, predicted noise, error logits, and optionally features.
        """
        t = t.float().view(-1, 1)
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)
        char_features, vt_features, context_features = self.get_context_embedding(
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, context_labels
        )
        attn_mask = (~attention_mask.bool()) if attention_mask is not None else None
        corrupt_label_embeddings = corrupt_label_embeddings + time_features

        # Cross-attention for visual-textual features
        vt_label_features = self.vt_label_attn(
            query=vt_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        )
        vt_char_features = self.vt_char_attn(
            query=vt_features, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else vt_features
        vt_context_features = self.vt_context_attn(
            query=vt_features, key=context_features, value=context_features, mask=attn_mask
        ) if context_features is not None else vt_features

        # Cross-attention for label features
        label_vt_features = self.label_vt_attn(
            query=corrupt_label_embeddings, key=vt_features, value=vt_features, mask=attn_mask
        )
        label_char_features = self.label_char_attn(
            query=corrupt_label_embeddings, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else corrupt_label_embeddings
        label_context_features = self.label_context_attn(
            query=corrupt_label_embeddings, key=context_features, value=context_features, mask=attn_mask
        ) if context_features is not None else corrupt_label_embeddings
        
        # Normalize and combine features
        vt_label_features = self.norm_vt_label(vt_label_features + vt_features)
        vt_context_features = self.norm_vt_context(vt_context_features + vt_features) if context_features is not None else vt_features
        vt_char_features = self.norm_vt_char(vt_char_features + vt_features) if char_features is not None else vt_features

        label_vt_features = self.norm_label_vt(label_vt_features + corrupt_label_embeddings)
        label_char_features = self.norm_label_char(label_char_features + corrupt_label_embeddings) if char_features is not None else corrupt_label_embeddings
        label_context_features = self.norm_label_context(label_context_features + corrupt_label_embeddings) if context_features is not None else corrupt_label_embeddings
        
        label_features_comb = (label_vt_features + label_char_features + label_context_features) / (2 + (context_features is not None))
        vt_features_comb = (vt_label_features + vt_char_features + vt_context_features) / (2 + (context_features is not None))
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1)
        
        # Predict outputs
        features = self.dropout(features)
        recon_emissions = self.fc(features)
        predicted_noise = self.noise_pred(features)
        error_pred_logits = self.error_pred(features).squeeze(-1)

        if return_features:
            return recon_emissions, predicted_noise, error_pred_logits, features
        return recon_emissions, predicted_noise, error_pred_logits

    def forward(self, labels=None, targets_noise=None, char_input_ids=None, input_ids=None, 
                attention_mask=None, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, 
                mode='pretrain', epoch=0, error_loss_weight=1.0):
        """Compute forward pass for pretraining or fine-tuning, returning loss and predictions.
        
        Args:
            labels (torch.Tensor): Clean NER labels (targets_unk for pretrain, targets_new for finetune).
            targets_noise (torch.Tensor): Corrupted NER labels (targets_noise for pretrain, targets_unk for finetune).
            char_input_ids (torch.Tensor, optional): Character input IDs.
            input_ids (torch.Tensor): Token input IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            images (torch.Tensor, optional): Main images.
            aux_imgs (torch.Tensor, optional): Auxiliary images.
            rcnn_imgs (torch.Tensor, optional): RCNN images.
            mode (str): 'pretrain' or 'finetune'.
            epoch (int): Current epoch number.
            error_loss_weight (float): Weight for error detection loss.
        
        Returns:
            tuple: Total loss, NER logits, error logits, error mask.
        """
        self.current_epoch = epoch
        bsz = input_ids.size(0) if input_ids is not None else labels.size(0)
        assert attention_mask is None or (attention_mask.max() <= 1 and attention_mask.min() >= 0), "Invalid attention_mask"

        # Scale error loss weight by epoch
        error_loss_weight = error_loss_weight * (2.0 + 0.5 * min(epoch / self.args.num_epochs, 1.0))

        # Validate inputs
        if labels is None or targets_noise is None:
            return None, None, None, None

        # Sample random timesteps
        t = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        
        # Corrupt labels and compute error mask
        corrupt_label_embeddings, noise, error_true_mask = self.corrupt(t, labels, targets_noise, attention_mask, mode=mode)
        
        # Set context_labels: noisy labels with errors to guide denoising
        # - Pretrain: targets_noise (corrupted labels with injected errors)
        # - Finetune: targets_noise (mapped to targets_unk, labels with annotation errors, in ddpm_train.py)
        context_labels = targets_noise
        
        # Denoise with noisy labels as context
        recon_emissions, predicted_noise, error_pred_logits, features = self.denoise(
            corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=context_labels,
            return_features=True
        )

        # Compute losses
        mse_loss = F.mse_loss(predicted_noise, noise)
        denoise_ce_loss = F.cross_entropy(
            recon_emissions.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=0,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        denoise_ce_loss = denoise_ce_loss.view(bsz, -1) * (1.0 + 2.0 * error_true_mask) * attention_mask
        valid_lengths = attention_mask.sum(dim=1).clamp(min=1)
        denoise_ce_loss = denoise_ce_loss.sum(dim=1) / valid_lengths
        denoise_ce_loss = denoise_ce_loss.mean() * error_loss_weight
        
        # Focal loss for sparse error detection
        alpha = 0.25  # Weight for positive class
        gamma = 2.0   # Focusing parameter
        bce = F.binary_cross_entropy_with_logits(error_pred_logits, error_true_mask, reduction='none')
        pt = torch.exp(-bce)
        error_focal_loss = (alpha * (1 - pt) ** gamma * bce).mean()
        
        # Alignment loss
        pred_labels = torch.argmax(recon_emissions, dim=-1)
        emission_error_mask = (pred_labels != labels).float()
        error_pred_mask = torch.sigmoid(error_pred_logits)
        alignment_loss = F.binary_cross_entropy(error_pred_mask, emission_error_mask, reduction='mean')
        
        # Combine losses
        if mode == 'pretrain':
            contrastive_loss = self.compute_contrastive_loss(features, error_true_mask, attention_mask)
            loss = 0.5 * mse_loss + 0.3 * denoise_ce_loss + 0.7 * error_focal_loss + 0.3 * contrastive_loss + getattr(self.args, 'alignment_loss_weight', 0.5) * alignment_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=self.args.device)
            loss = 0.5 * mse_loss + 0.5 * denoise_ce_loss + 0.7 * error_focal_loss + getattr(self.args, 'alignment_loss_weight', 0.5) * alignment_loss

        # Store individual losses
        self.mse_loss = mse_loss
        self.denoise_ce_loss = denoise_ce_loss
        self.error_focal_loss = error_focal_loss
        self.contrastive_loss = contrastive_loss
        self.alignment_loss = alignment_loss

        return loss, recon_emissions, error_pred_logits, error_true_mask

    def reverse_diffusion(self, char_input_ids, input_ids, attention_mask, token_type_ids, 
                         images, aux_imgs, rcnn_imgs, context_labels, steps=20, guidance_scale=5.0, temperature=0.8):
        """Perform reverse diffusion to generate corrected labels and error mask.
        
        Args:
            char_input_ids (torch.Tensor): Character input IDs.
            input_ids (torch.Tensor): Token input IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            images (torch.Tensor, optional): Main images.
            aux_imgs (torch.Tensor, optional): Auxiliary images.
            rcnn_imgs (torch.Tensor, optional): RCNN images.
            context_labels (torch.Tensor): Noisy NER labels used as context.
            steps (int): Number of reverse diffusion steps.
            guidance_scale (float): Classifier-free guidance scale.
            temperature (float): Temperature for scaling logits.
        
        Returns:
            tuple: Predicted NER labels and error mask.
        """
        batch_size, seq_len = input_ids.shape
        label_embeddings = torch.randn(batch_size, seq_len, self.args.label_hidden_dim, device=self.args.device)

        step_sizes = torch.linspace(1.0, 0.1, steps, device=self.args.device)
        t_values = torch.linspace(steps - 1, 0, steps, device=self.args.device).long()

        for i, t in enumerate(t_values):
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            recon_emissions_cond, predicted_noise_cond, error_logits_cond, _ = self.denoise(
                label_embeddings, t_tensor, char_input_ids, input_ids, attention_mask,
                token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=context_labels,
                return_features=True
            )
            recon_emissions_uncond, predicted_noise_uncond, error_logits_uncond, _ = self.denoise(
                label_embeddings, t_tensor, char_input_ids, input_ids, attention_mask,
                token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=None,
                return_features=True
            )
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond) / temperature

            alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1, 1)
            alpha_t = self.noise_scheduler.alpha[t].view(-1, 1, 1)
            sigma_t = torch.sqrt(1 - alpha_bar_t) * torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_bar_t)
            coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            label_embeddings = (label_embeddings - coeff * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                z = torch.randn_like(label_embeddings) * step_sizes[i]
                label_embeddings += sigma_t * z

        recon_emissions, _, error_pred_logits, _ = self.denoise(
            label_embeddings, torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs,
            context_labels=context_labels, return_features=True
        )
        diffusion_logits = recon_emissions / temperature
        error_pred_mask = torch.sigmoid(error_pred_logits)
        return diffusion_logits.argmax(dim=-1), error_pred_mask