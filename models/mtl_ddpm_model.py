import torch
import os
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
from .char_lstm import CharLSTM
from .bert_model import HMNeTNERModel
from .unimo_model import UnimoCRFModel
from utils.attention import MultiAttn, PositionalEncoding

# Manages the noise schedule for the diffusion process
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu", schedule_type="cosine"):
        """Initialize noise scheduler with cosine or linear schedule."""
        self.timesteps = timesteps
        self.device = device
        assert 0 < beta_start < beta_end < 1, "Invalid beta range"
        
        # Set up cosine noise schedule (default) or linear schedule
        if schedule_type == "cosine":
            t = torch.linspace(0, 1, timesteps + 1, device=device)[:-1]
            self.beta = 1 - torch.cos(t * torch.pi / 2) ** 2
            self.beta = self.beta * (beta_end - beta_start) + beta_start
        else:
            self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
            
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.noise_scale = 1.0

    def set_noise_scale(self, scale):
        """Set the scale for noise addition."""
        self.noise_scale = scale

    def add_noise(self, x, t, attention_mask=None):
        """Add noise to input embeddings at timestep t, respecting attention mask."""
        assert t.max() < self.timesteps and t.min() >= 0, "Invalid timestep"
        noise = torch.randn_like(x) * self.noise_scale
        signal_rate_t = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        noise_rate_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        noisy_x = signal_rate_t * x + noise_rate_t * noise
        # Apply noise only to valid (non-padded) tokens if mask provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            noisy_x = mask * noisy_x + (1 - mask) * x
        return noisy_x, noise

# Diffusion model for error detection in NER labels
class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_embedding_table=None, clstm_path=None, ner_model_name="hvpnet"):
        """Initialize the diffusion model with label, character, and visual-textual encoders."""
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.ner_model_name = ner_model_name
        
        # Time embedding for diffusion steps
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
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
        
        # Cross-attention layers for integrating label, character, and visual-textual features
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
        self.label_unk_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=self.args.label_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.char_vt_attn = MultiAttn(
            query_dim=self.args.char_hidden_dim,
            key_dim=vt_hidden_size,
            value_dim=vt_hidden_size,
            emb_dim=self.args.char_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.char_label_attn = MultiAttn(
            query_dim=self.args.char_hidden_dim,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=self.args.char_hidden_dim,
            num_heads=4, 
            dropout_rate=0.4
        )
        self.char_unk_attn = MultiAttn(
            query_dim=self.args.char_hidden_dim,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=self.args.char_hidden_dim,
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
        self.vt_unk_attn = MultiAttn(
            query_dim=vt_hidden_size,
            key_dim=self.args.label_hidden_dim,
            value_dim=self.args.label_hidden_dim,
            emb_dim=vt_hidden_size,
            num_heads=4, 
            dropout_rate=0.4
        )
        
        # Normalization layers for stabilizing attention outputs
        self.norm_char_label = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_label = nn.LayerNorm(self.vt_hidden_size)
        self.norm_label_vt = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_char = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_unk = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_char_vt = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_char_unk = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_char = nn.LayerNorm(self.vt_hidden_size)
        self.norm_vt_unk = nn.LayerNorm(self.vt_hidden_size)
        
        # Output layers: predict labels, noise, and error mask
        self.fc = nn.Sequential(
            nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, num_labels),
            nn.LayerNorm(num_labels)
        )
        self.noise_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, self.args.label_hidden_dim)
        self.error_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, 1)  # Binary error mask prediction
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_labels, batch_first=True)
        self.label_smoothing = 0.1
        
        # Valid labels for flipping (exclude "O", adjust based on label_map)
        self.valid_labels = torch.tensor([i for i in range(num_labels) if i not in [0]], device=args.device)

    def get_label_embedding(self, labels, attention_mask=None):
        """Convert label indices to embeddings with positional encoding and self-attention."""
        assert labels is not None, "labels required"
        assert labels.max() < self.label_embedding_table.shape[0], "Label indices out of range"
        label_features = self.label_embedding_table[labels]
        label_features = self.label_mlp(label_features)
        label_features = self.label_pos_encoder(label_features)
        label_mask = 1 - attention_mask if attention_mask is not None else None
        label_features = self.label_self_attn(
            query=label_features, key=label_features, value=label_features, mask=label_mask
        )
        return label_features

    def get_context_embedding(self, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, 
                             images=None, aux_imgs=None, rcnn_imgs=None, context_labels=None):
        """Generate embeddings for characters, visual-textual input, and context labels."""
        bsz = input_ids.size(0)
        
        # Character embeddings using CharLSTM
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
            attn_mask = 1 - attention_mask if attention_mask is not None else None
            char_features = self.char_self_attn(char_features, char_features, char_features, mask=attn_mask)
        else:
            char_features = None
        
        # Visual-textual embeddings using HVPNet or MKGFormer
        if self.ner_model_name == "hvpnet":
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                  pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
            vt_features = out.last_hidden_state
        assert vt_features.shape == (bsz, self.args.max_seq_len, self.vt_hidden_size), "VT output shape mismatch"
        
        # Context label embeddings
        context_features = None
        if context_labels is not None:
            context_features = self.get_label_embedding(context_labels, attention_mask)
        
        return char_features, vt_features, context_features

    def inject_random_noise(self, labels, attention_mask):
        """Randomly corrupt labels (omit, flip, swap) for pretraining context, respecting attention mask."""
        bsz, seq_len = labels.shape
        corrupted_labels = labels.clone()
        error_mask = torch.zeros_like(labels, dtype=torch.float)
        
        for b in range(bsz):
            valid_positions = attention_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) < 2:
                continue
            num_corruptions = max(1, int(self.args.noise_rate * len(valid_positions)))
            corrupt_positions = valid_positions[torch.randperm(len(valid_positions))[:num_corruptions]]
            
            for pos in corrupt_positions:
                corruption_type = torch.multinomial(torch.tensor([0.3, 0.4, 0.3], device=self.args.device), 1).item()
                
                if corruption_type == 0:  # Omit: set to "O" (index 0)
                    corrupted_labels[b, pos] = 0
                    error_mask[b, pos] = 1.0
                elif corruption_type == 1:  # Flip: set to random valid label
                    new_label = self.valid_labels[torch.randint(0, len(self.valid_labels), (1,), device=self.args.device)]
                    corrupted_labels[b, pos] = new_label
                    error_mask[b, pos] = 1.0
                elif corruption_type == 2 and pos + 1 < seq_len and attention_mask[b, pos + 1]:  # Swap: swap with next valid token
                    corrupted_labels[b, pos], corrupted_labels[b, pos + 1] = corrupted_labels[b, pos + 1], corrupted_labels[b, pos]
                    error_mask[b, pos] = 1.0
                    error_mask[b, pos + 1] = 1.0
        
        return corrupted_labels, error_mask

    def compute_contrastive_loss(self, features, error_mask, attention_mask):
        """Compute contrastive loss to distinguish erroneous from correct tokens."""
        bsz, seq_len, dim = features.shape
        features = features.view(-1, dim)
        error_mask = error_mask.view(-1)
        mask = attention_mask.view(-1).bool()
        
        features = features[mask]
        error_mask = error_mask[mask]
        
        features = F.normalize(features, dim=-1)
        sim_matrix = torch.matmul(features, features.t())
        label_matrix = (error_mask.unsqueeze(0) == error_mask.unsqueeze(1)).float()
        temperature = 0.07
        sim_matrix = sim_matrix / temperature
        pos_sim = sim_matrix * label_matrix
        neg_sim = sim_matrix * (1 - label_matrix)
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        loss = -torch.log(pos_exp.sum(dim=-1) / (pos_exp.sum(dim=-1) + neg_exp.sum(dim=-1) + 1e-8))
        return loss.mean()

    def corrupt(self, t, labels, attention_mask, mode='pretrain'):
        """Corrupt labels with diffusion noise; generate noisy context for pretraining."""
        if mode == 'pretrain':
            # Generate targets_noise for context
            targets_noise, error_mask = self.inject_random_noise(labels, attention_mask)
            # Corrupt targets_unk directly with diffusion noise
            label_features = self.get_label_embedding(labels, attention_mask)
            corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
            return corrupt_label_embeddings, noise, error_mask, targets_noise
        else:
            # Corrupt labels (targets_new in fine-tuning) with diffusion noise
            label_features = self.get_label_embedding(labels, attention_mask)
            corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
            return corrupt_label_embeddings, noise, None, None

    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, context_labels=None, return_features=False):
        """Denoise corrupted embeddings using noisy label context, predicting labels, noise, and error mask."""
        t = t.float().view(-1, 1)
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)
        char_features, vt_features, context_features = self.get_context_embedding(
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, context_labels
        )
        attn_mask = 1 - attention_mask if attention_mask is not None else None
        corrupt_label_embeddings = corrupt_label_embeddings + time_features

        # Apply cross-attention to integrate features
        char_label_features = self.char_label_attn(
            query=char_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        ) if char_features is not None else char_features
        char_context_features = self.char_unk_attn(
            query=char_features, key=context_features, value=context_features, mask=attn_mask
        ) if char_features is not None and context_features is not None else char_features
        vt_label_features = self.vt_label_attn(
            query=vt_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        )
        vt_context_features = self.vt_unk_attn(
            query=vt_features, key=context_features, value=context_features, mask=attn_mask
        ) if context_features is not None else vt_features
        label_vt_features = self.label_vt_attn(
            query=corrupt_label_embeddings, key=vt_features, value=vt_features, mask=attn_mask
        )
        label_char_features = self.label_char_attn(
            query=corrupt_label_embeddings, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else corrupt_label_embeddings
        label_context_features = self.label_unk_attn(
            query=corrupt_label_embeddings, key=context_features, value=context_features, mask=attn_mask
        ) if context_features is not None else corrupt_label_embeddings
        char_vt_features = self.char_vt_attn(
            query=char_features, key=vt_features, value=vt_features, mask=attn_mask
        ) if char_features is not None else char_features
        vt_char_features = self.vt_char_attn(
            query=vt_features, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else vt_features

        # Normalize and add residual connections
        char_label_features = self.norm_char_label(char_label_features + char_features) if char_features is not None else char_features
        char_context_features = self.norm_char_unk(char_context_features + char_features) if char_features is not None and context_features is not None else char_features
        vt_label_features = self.norm_vt_label(vt_label_features + vt_features)
        vt_context_features = self.norm_vt_unk(vt_context_features + vt_features) if context_features is not None else vt_features
        label_vt_features = self.norm_label_vt(label_vt_features + corrupt_label_embeddings)
        label_char_features = self.norm_label_char(label_char_features + corrupt_label_embeddings) if char_features is not None else corrupt_label_embeddings
        label_context_features = self.norm_label_unk(label_context_features + corrupt_label_embeddings) if context_features is not None else corrupt_label_embeddings
        char_vt_features = self.norm_char_vt(char_vt_features + char_features) if char_features is not None else char_features
        vt_char_features = self.norm_vt_char(vt_char_features + vt_features) if char_features is not None else vt_features

        # Combine features for output
        label_features_comb = (label_vt_features + label_char_features + label_context_features) / (2 + (context_features is not None))
        vt_features_comb = (vt_label_features + vt_char_features + vt_context_features) / (2 + (context_features is not None))
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1)
        
        features = self.dropout(features)
        recon_emissions = self.fc(features)
        predicted_noise = self.noise_pred(features)
        error_logits = self.error_pred(features).squeeze(-1)  # Binary logits for error mask

        if return_features:
            return recon_emissions, predicted_noise, error_logits, features
        return recon_emissions, predicted_noise, error_logits

    def forward(self, labels=None, targets_unk=None, targets_new=None, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, mode='pretrain', epoch=0, error_loss_weight=1.0):
        """Compute forward pass for pretraining or fine-tuning, returning loss and predictions."""
        self.current_epoch = epoch
        bsz = input_ids.size(0) if input_ids is not None else labels.size(0)
        assert attention_mask is None or (attention_mask.max() <= 1 and attention_mask.min() >= 0), "Invalid attention_mask"

        # Scale error loss weight based on epoch
        error_loss_weight = error_loss_weight * (1.0 + 0.5 * min(epoch / self.args.num_epochs, 1.0))

        if labels is None and (mode == 'pretrain' or targets_unk is None or targets_new is None):
            return None, None

        # Randomly sample timesteps
        t = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        if mode == 'pretrain':
            corrupt_label_embeddings, noise, error_mask, targets_noise = self.corrupt(t, labels, attention_mask, mode=mode)
            target_labels = labels  # Correct labels for pretraining
            context_labels = targets_noise
        else:
            corrupt_label_embeddings, noise, _, _ = self.corrupt(t, targets_new, attention_mask, mode=mode)
            error_mask = (targets_unk != targets_new).float()  # Error mask for fine-tuning
            target_labels = targets_new
            context_labels = targets_unk

        # Denoise and predict
        recon_emissions, predicted_noise, error_logits, features = self.denoise(
            corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=context_labels,
            return_features=True
        )

        # Compute losses
        mse_loss = F.mse_loss(predicted_noise, noise)
        denoise_crf_loss = -self.crf(recon_emissions, target_labels, mask=attention_mask.bool(), 
                                   reduction='none', label_smoothing=self.label_smoothing)
        denoise_crf_loss = (denoise_crf_loss * (1.0 + 8.0 * error_mask)).mean() * error_loss_weight
        error_bce_loss = F.binary_cross_entropy_with_logits(error_logits, error_mask, reduction='mean')
        
        if mode == 'pretrain':
            contrastive_loss = self.compute_contrastive_loss(features, error_mask, attention_mask)
            loss = 0.5 * mse_loss + 0.3 * denoise_crf_loss + 0.3 * error_bce_loss + 0.2 * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=self.args.device)
            loss = 0.3 * mse_loss + 2.0 * denoise_crf_loss + 0.5 * error_bce_loss

        # Store individual losses for logging
        self.mse_loss = mse_loss
        self.denoise_crf_loss = denoise_crf_loss
        self.error_bce_loss = error_bce_loss
        self.contrastive_loss = contrastive_loss
        self.kl_loss = torch.tensor(0.0, device=self.args.device)

        return loss, recon_emissions, error_logits

    def reverse_diffusion(self, char_input_ids, input_ids, attention_mask, token_type_ids, 
                         images, aux_imgs, rcnn_imgs, context_labels, steps=20, guidance_scale=5.0, temperature=0.8):
        """Perform reverse diffusion to generate corrected labels and error mask."""
        batch_size, seq_len = input_ids.shape
        label_embeddings = torch.randn(batch_size, seq_len, self.args.label_hidden_dim, device=self.args.device)

        step_sizes = torch.linspace(1.0, 0.1, steps, device=self.args.device)
        t_values = torch.linspace(steps - 1, 0, steps, device=self.args.device).long()

        for i, t in enumerate(t_values):
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            # Conditional prediction with context
            recon_emissions_cond, predicted_noise_cond, error_logits_cond, _ = self.denoise(
                label_embeddings, t_tensor, char_input_ids, input_ids, attention_mask,
                token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=context_labels,
                return_features=True
            )
            # Unconditional prediction
            recon_emissions_uncond, predicted_noise_uncond, error_logits_uncond, _ = self.denoise(
                label_embeddings, t_tensor, char_input_ids, input_ids, attention_mask,
                token_type_ids, images, aux_imgs, rcnn_imgs, context_labels=None,
                return_features=True
            )
            # Apply classifier-free guidance
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond) / temperature

            # Update embeddings
            alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1, 1)
            alpha_t = self.noise_scheduler.alpha[t].view(-1, 1, 1)
            sigma_t = torch.sqrt(1 - alpha_bar_t) * torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_bar_t)
            coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            label_embeddings = (label_embeddings - coeff * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                z = torch.randn_like(label_embeddings) * step_sizes[i]
                label_embeddings += sigma_t * z

        # Final denoising step
        recon_emissions, _, error_logits, _ = self.denoise(
            label_embeddings, torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs,
            context_labels=context_labels, return_features=True
        )
        diffusion_logits = recon_emissions / temperature
        error_mask = torch.sigmoid(error_logits) > 0.5
        return diffusion_logits.argmax(dim=-1), error_mask