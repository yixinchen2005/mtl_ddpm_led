import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert_model import HMNeTNERModel
from .unimo_model import UnimoCRFModel
from .gnn_model import HeteroLabelEmbeddingGNN
from utils.attention import UnifiedAttention
import os
import logging

logger = logging.getLogger(__name__)

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
    
    def step(self, pred_clean, t, noisy_x):
        """Perform a denoising step using predicted clean embeddings."""
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        alpha_bar_t_prev = self.alpha_bar[t-1].view(-1, 1, 1) if (t-1).any() >= 0 else torch.ones_like(alpha_bar_t)
        noise_denom = torch.sqrt(1 - alpha_bar_t)
        noise_denom = torch.where(noise_denom == 0, torch.ones_like(noise_denom) * 1e-8, noise_denom)
        eps = (noisy_x - torch.sqrt(alpha_bar_t) * pred_clean) / noise_denom
        x = torch.sqrt(alpha_bar_t_prev) * pred_clean + torch.sqrt(1 - alpha_bar_t_prev) * eps
        return x

# FiLM Layer for Timestep Conditioning
class FiLM(nn.Module):
    def __init__(self, dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
    
    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * gamma + beta

# Optional Post-Processing for Valid BIO Sequences
def post_process_bio_labels(bio_labels, num_bio_labels):
    """Enforce valid BIO transitions (e.g., no I-X without B-X)."""
    batch_size, seq_len = bio_labels.shape
    valid_labels = bio_labels.clone()
    for b in range(batch_size):
        for i in range(seq_len):
            curr_label = valid_labels[b, i]
            if curr_label in [1, 3, 5]:  # I-PER, I-LOC, I-ORG
                prev_label = valid_labels[b, i-1] if i > 0 else None
                corresponding_b = curr_label - 1  # B-PER, B-LOC, B-ORG
                if prev_label not in [corresponding_b, curr_label]:
                    valid_labels[b, i] = 6  # Relabel as O
    return valid_labels

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_embeddings=None, gnn_path=None, ner_model_name="hvpnet"):
        """Initialize the diffusion model for NER pre-training."""
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.ner_model_name = ner_model_name

        # Noise Scheduler
        self.noise_scheduler = NoiseScheduler(
            timesteps=self.args.train_steps, 
            device=self.args.device,
            schedule_type="cosine"
        )
        
        # Time embedding MLP for diffusion steps
        self.time_proj = nn.Linear(1, self.args.embed_dim)
        
        # Label encoder
        self.label_encoder = HeteroLabelEmbeddingGNN(
            label_embeddings=label_embeddings, 
            hidden_dim=self.args.label_hidden_dim, 
            num_labels=num_labels
        ).to(self.args.device)
        if gnn_path:
            logger.info(f"Loading GNN weights from {os.path.join(gnn_path, 'gnn_hetero_best_decoder.pth')}")
            self.label_encoder.load_state_dict(torch.load(os.path.join(gnn_path, "gnn_hetero_best_decoder.pth")))
        self.label_proj = nn.Linear(self.args.label_hidden_dim, self.args.embed_dim)
        self.label_self_attn = nn.ModuleList([
            UnifiedAttention(
                query_dim=self.args.embed_dim, 
                key_dim=self.args.embed_dim, 
                value_dim=self.args.embed_dim, 
                emb_dim=self.args.embed_dim, 
                num_heads=min(4, self.args.embed_dim // 16),
                dropout_rate=0.1,
                use_relative=True,
                max_len=self.args.max_seq_len
            ) for _ in range(3)  # Stack 3 self-attention layers
        ])
        self.label_self_attn_dropout = nn.Dropout(0.2)
        
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
        self.vt_encoder = self.vt_encoder.to(self.args.device)
        self.vt_proj = nn.Linear(vt_hidden_size, self.args.embed_dim)
        
        # Cross-attention layers
        self.label_vt_attn = nn.ModuleList([
            UnifiedAttention(
                query_dim=self.args.embed_dim,
                key_dim=self.args.embed_dim,
                value_dim=self.args.embed_dim,
                emb_dim=self.args.embed_dim,
                num_heads=min(4, self.args.embed_dim // 16),
                dropout_rate=0.2,
                use_relative=False,
                max_len=self.args.max_seq_len
            ) for _ in range(2)  # Stack 2 cross-attention layers
        ])
        self.vt_label_attn = nn.ModuleList([
            UnifiedAttention(
                query_dim=self.args.embed_dim,
                key_dim=self.args.embed_dim,
                value_dim=self.args.embed_dim,
                emb_dim=self.args.embed_dim,
                num_heads=min(4, self.args.embed_dim // 16),
                dropout_rate=0.2,
                use_relative=False,
                max_len=self.args.max_seq_len
            ) for _ in range(2)  # Stack 2 cross-attention layers
        ])
        self.cross_attn_dropout = nn.Dropout(0.2)
        
        # Normalization and FiLM
        self.norm_label_vt = nn.LayerNorm(self.args.embed_dim)
        self.norm_vt_label = nn.LayerNorm(self.args.embed_dim)
        self.film = FiLM(self.args.embed_dim)
        
        # Output layers
        self.embedding_pred = nn.Sequential(
            nn.Linear(2 * self.args.embed_dim, self.args.embed_dim),
            nn.GELU()
        )
        self.classifier = nn.Linear(self.args.embed_dim, num_labels)
        self.output_dropout = nn.Dropout(0.3)

    def get_label_embedding(self, labels, attention_mask=None):
        """Convert label indices to embeddings with GNN and stacked self-attention."""
        assert labels is not None, "labels required"
        labels = labels.to(self.args.device)
        if labels.max() >= self.num_labels:
            logger.warning(f"Label indices out of range: max={labels.max().item()}, num_labels={self.num_labels}")
        _, label_features, _ = self.label_encoder(label_indices=labels)
        label_features = self.label_proj(label_features)
        label_mask = attention_mask.bool() if attention_mask is not None else None
        for self_attn in self.label_self_attn:
            label_features = self_attn(
                query=label_features, key=label_features, value=label_features, 
                mask=label_mask
            )
            label_features = self.label_self_attn_dropout(label_features)
        return label_features

    def get_context_embedding(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                             images=None, aux_imgs=None, rcnn_imgs=None):
        """Generate embeddings for visual-textual input."""
        assert input_ids is not None, "input_ids required"
        input_ids = input_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(self.args.device) if token_type_ids is not None else None
        images = images.to(self.args.device) if images is not None else None
        aux_imgs = aux_imgs.to(self.args.device) if aux_imgs is not None else None
        rcnn_imgs = rcnn_imgs.to(self.args.device) if rcnn_imgs is not None else None
        
        if self.ner_model_name == "hvpnet":
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                pixel_values=images, 
                aux_values=aux_imgs, 
                rcnn_values=rcnn_imgs, 
                return_dict=True
            )
            vt_features = out.last_hidden_state
        vt_features = self.vt_proj(vt_features)
        assert vt_features.shape == (input_ids.size(0), self.args.max_seq_len, self.args.embed_dim), "VT output shape mismatch"
        
        return vt_features

    def corrupt(self, t, labels, attention_mask):
        """Corrupt labels with diffusion noise."""
        label_features = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, _ = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        return corrupt_label_embeddings, label_features

    def denoise(self, corrupt_label_embeddings, t, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Denoise corrupted embeddings, returning predicted clean embeddings."""
        t = t.float().view(-1, 1).to(self.args.device)
        time_features = torch.sin(self.time_proj(t)).unsqueeze(1)  # [B, 1, embed_dim]
        vt_features = self.get_context_embedding(
            input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs
        )
        corrupt_label_embeddings = corrupt_label_embeddings + time_features
        attn_mask = attention_mask.bool() if attention_mask is not None else None

        label_vt_features = corrupt_label_embeddings
        for attn in self.label_vt_attn:
            label_vt_features = attn(
                query=label_vt_features, key=vt_features, value=vt_features, 
                mask=attn_mask
            )
            label_vt_features = self.cross_attn_dropout(label_vt_features)
        
        vt_label_features = vt_features
        for attn in self.vt_label_attn:
            vt_label_features = attn(
                query=vt_label_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, 
                mask=attn_mask
            )
            vt_label_features = self.cross_attn_dropout(vt_label_features)
        
        vt_label_features = self.norm_vt_label(vt_label_features + vt_features)
        label_vt_features = self.norm_label_vt(label_vt_features + corrupt_label_embeddings)
        
        features = torch.cat((label_vt_features, vt_label_features), dim=-1)
        features = self.output_dropout(features)
        pred_embeddings = self.film(self.embedding_pred(features), time_features)
        
        return pred_embeddings

    def forward(self, labels=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Compute forward pass for pre-training, returning loss and NER logits."""
        bsz = input_ids.size(0) if input_ids is not None else labels.size(0)
        assert attention_mask is None or (attention_mask.max() <= 1 and attention_mask.min() >= 0), "Invalid attention_mask"
        assert labels is not None, "labels required"

        # Sample timesteps with args.t_zero_prob probability for t=0
        t_random = torch.where(
            torch.rand(bsz, device=self.args.device) < self.args.t_zero_prob,
            torch.zeros(bsz, device=self.args.device, dtype=torch.long),
            torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        )
        corrupt_label_embeddings, clean_embeddings = self.corrupt(t_random, labels, attention_mask)
        pred_embeddings = self.denoise(
            corrupt_label_embeddings, t_random, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs
        )
        mse_loss = F.mse_loss(pred_embeddings, clean_embeddings)
        
        # Compute CE loss only when t_random = 0
        ce_loss = torch.tensor(0.0, device=self.args.device)
        logits = None
        t_zero_mask = (t_random == 0)
        if t_zero_mask.any():
            # Select pred_embeddings for t=0
            valid_pred_embeddings = pred_embeddings[t_zero_mask]
            valid_labels = labels[t_zero_mask]
            logits = torch.zeros(bsz, labels.size(1), self.num_labels, device=self.args.device)
            logits[t_zero_mask] = self.classifier(valid_pred_embeddings)
            # Compute CE loss with reduction='mean' over valid tokens
            valid_logits = logits[t_zero_mask].view(-1, self.num_labels)
            valid_labels = valid_labels.view(-1)
            ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
        
        # Combine losses
        loss = mse_loss + self.args.ce_weight * ce_loss

        self.mse_loss = mse_loss
        self.ce_loss = ce_loss

        return loss, logits

    def reverse_diffusion(self, input_ids, attention_mask, token_type_ids, 
                         images, aux_imgs, rcnn_imgs, steps=50, post_process=False):
        """Perform reverse diffusion to generate NER labels."""
        batch_size, seq_len = input_ids.shape
        label_embeddings = torch.randn(batch_size, seq_len, self.args.embed_dim, device=self.args.device)
        
        for t in reversed(range(steps)):
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            pred_embeddings = self.denoise(
                label_embeddings, t=t_tensor, input_ids=input_ids, 
                attention_mask=attention_mask, token_type_ids=token_type_ids, images=images, 
                aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs
            )
            label_embeddings = self.noise_scheduler.step(pred_embeddings, t_tensor, label_embeddings)
        
        pred_embeddings = self.denoise(
            label_embeddings, t=torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            input_ids=input_ids, attention_mask=attention_mask, 
            token_type_ids=token_type_ids, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs
        )
        
        logits = self.classifier(pred_embeddings)
        pred_labels = torch.argmax(logits, dim=-1)
        if post_process:
            pred_labels = post_process_bio_labels(pred_labels, self.num_labels)
        return pred_labels