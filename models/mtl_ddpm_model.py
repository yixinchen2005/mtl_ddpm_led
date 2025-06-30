import torch
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
from .char_lstm import CharLSTM
from .bert_model import HMNeTNERModel
from .unimo_model import UnimoCRFModel
from utils.attention import MultiAttn, PositionalEncoding
import os

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu", schedule_type="cosine"):
        """Initialize noise scheduler with cosine or linear schedule."""
        self.timesteps = timesteps
        self.device = device
        assert 0 < beta_start < beta_end < 1, "Invalid beta range"
        
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
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            noisy_x = mask * noisy_x + (1 - mask) * x
        return noisy_x, noise

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_embedding_table=None, clstm_path=None, ner_model_name="hvpnet"):
        """Initialize the diffusion model for NER pre-training."""
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.ner_model_name = ner_model_name
        
        # Time embedding MLP for diffusion steps
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
        self.noise_scheduler = NoiseScheduler(
            timesteps=self.args.train_steps, 
            device=self.args.device,
            schedule_type="cosine"
        )
        
        # Label encoder
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
        
        # Character LSTM encoder
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
        
        # Visual-textual encoder
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
        
        # Cross-attention layers
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
        
        # Normalization layers
        self.norm_label_vt = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_char = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_vt_label = nn.LayerNorm(self.vt_hidden_size)
        self.norm_vt_char = nn.LayerNorm(self.vt_hidden_size)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Output layers
        self.fc = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, num_labels)
        self.noise_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, self.args.label_hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def get_label_embedding(self, labels, attention_mask=None):
        """Convert label indices to embeddings with positional encoding and self-attention."""
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
                             images=None, aux_imgs=None, rcnn_imgs=None):
        """Generate embeddings for characters and visual-textual input."""
        assert input_ids is not None, "input_ids required"
        bsz = input_ids.size(0)
        
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
        
        if self.ner_model_name == "hvpnet":
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                  pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
            vt_features = out.last_hidden_state
        assert vt_features.shape == (bsz, self.args.max_seq_len, self.vt_hidden_size), "VT output shape mismatch"
        
        return char_features, vt_features

    def corrupt(self, t, labels, attention_mask):
        """Corrupt labels with diffusion noise."""
        label_features = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        return corrupt_label_embeddings, noise

    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Denoise corrupted embeddings, predicting noise and NER logits."""
        t = t.float().view(-1, 1)
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)
        char_features, vt_features = self.get_context_embedding(
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs
        )
        attn_mask = (~attention_mask.bool()) if attention_mask is not None else None
        corrupt_label_embeddings = corrupt_label_embeddings + time_features

        vt_label_features = self.vt_label_attn(
            query=vt_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        )
        vt_char_features = self.vt_char_attn(
            query=vt_features, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else vt_features

        label_vt_features = self.label_vt_attn(
            query=corrupt_label_embeddings, key=vt_features, value=vt_features, mask=attn_mask
        )
        label_char_features = self.label_char_attn(
            query=corrupt_label_embeddings, key=char_features, value=char_features, mask=attn_mask
        ) if char_features is not None else corrupt_label_embeddings
        
        vt_label_features = self.norm_vt_label(vt_label_features + vt_features)
        vt_char_features = self.norm_vt_char(vt_char_features + vt_features) if char_features is not None else vt_features

        label_vt_features = self.norm_label_vt(label_vt_features + corrupt_label_embeddings)
        label_char_features = self.norm_label_char(label_char_features + corrupt_label_embeddings) if char_features is not None else corrupt_label_embeddings
        
        label_features_comb = (label_vt_features + label_char_features) / (1 + (char_features is not None))
        vt_features_comb = (vt_label_features + vt_char_features) / (1 + (char_features is not None))
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1)
        
        features = self.dropout(features)
        recon_emissions = self.fc(features)
        predicted_noise = self.noise_pred(features)
        return recon_emissions, predicted_noise

    def forward(self, labels=None, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Compute forward pass for pre-training, returning loss and NER logits."""
        bsz = input_ids.size(0) if input_ids is not None else labels.size(0)
        assert attention_mask is None or (attention_mask.max() <= 1 and attention_mask.min() >= 0), "Invalid attention_mask"
        assert labels is not None, "labels required"

        # Sample random timesteps
        t = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        
        # Corrupt labels (targets_unk)
        corrupt_label_embeddings, noise = self.corrupt(t, labels, attention_mask)
        
        # Denoise with visual-textual and character context
        recon_emissions, predicted_noise = self.denoise(
            corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs
        )

        # Compute losses
        mse_loss = F.mse_loss(predicted_noise, noise)
        crf_loss = -1 * self.crf(recon_emissions, labels, mask=attention_mask.bool(), reduction='mean')
        
        # Combine losses
        loss = 0.5 * mse_loss + 0.5 * crf_loss

        self.mse_loss = mse_loss
        self.crf_loss = crf_loss

        return loss, recon_emissions

    def reverse_diffusion(self, char_input_ids, input_ids, attention_mask, token_type_ids, 
                         images, aux_imgs, rcnn_imgs, steps=20, temperature=1.0):
        """Perform reverse diffusion to generate NER labels for pre-training."""
        batch_size, seq_len = input_ids.shape
        label_embeddings = torch.randn(batch_size, seq_len, self.args.label_hidden_dim, device=self.args.device)

        step_sizes = torch.linspace(1.0, 0.1, steps, device=self.args.device)
        t_values = torch.linspace(steps - 1, 0, steps, device=self.args.device).long()

        for i, t in enumerate(t_values):
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            _, predicted_noise = self.denoise(
                label_embeddings, t=t_tensor, char_input_ids=char_input_ids, input_ids=input_ids, 
                attention_mask=attention_mask, token_type_ids=token_type_ids, images=images, 
                aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs
            )

            alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1, 1)
            alpha_t = self.noise_scheduler.alpha[t].view(-1, 1, 1)
            sigma_t = torch.sqrt(1 - alpha_bar_t) * torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_bar_t)
            coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            label_embeddings = (label_embeddings - coeff * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                z = torch.randn_like(label_embeddings) * step_sizes[i]
                label_embeddings += sigma_t * z

        recon_emissions, _ = self.denoise(
            label_embeddings, t=torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            char_input_ids=char_input_ids, input_ids=input_ids, attention_mask=attention_mask, 
            token_type_ids=token_type_ids, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs
        )
        diffusion_logits = recon_emissions / temperature
        return diffusion_logits.argmax(dim=-1)