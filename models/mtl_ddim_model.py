import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.attention import UnifiedAttention
import os
import logging
import math

logger = logging.getLogger(__name__)

class DDIMScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu", schedule_type="cosine"):
        """Initialize DDIM scheduler with cosine schedule."""
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
        assert t.max() < self.timesteps and t.min() >= 0, f"Invalid timestep: t={t}"
        noise = torch.randn_like(x) * self.noise_scale
        signal_rate_t = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        noise_rate_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        noisy_x = signal_rate_t * x + noise_rate_t * noise
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            noisy_x = mask * noisy_x + (1 - mask) * x
        return noisy_x, noise
    
    def step(self, pred_clean, t, noisy_x, eta=0.0):
        """Perform a DDIM denoising step."""
        assert t.max() < self.timesteps and t.min() >= 0, f"Invalid timestep: t={t}"
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        alpha_bar_t_prev = self.alpha_bar[t-1].view(-1, 1, 1) if (t-1).min() >= 0 else torch.ones_like(alpha_bar_t)
        
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
        noise_denom = torch.sqrt(1 - alpha_bar_t)
        noise_denom = torch.where(noise_denom == 0, torch.ones_like(noise_denom) * 1e-5, noise_denom)
        eps = (noisy_x - torch.sqrt(alpha_bar_t) * pred_clean) / noise_denom
        
        x = (torch.sqrt(alpha_bar_t_prev) * pred_clean + 
             torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps + 
             sigma_t * torch.randn_like(noisy_x))
        return x

class FiLM(nn.Module):
    def __init__(self, dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
    
    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * gamma + beta

def post_process_bio_labels(bio_labels, num_bio_labels):
    """Enforce valid BIO transitions."""
    batch_size, seq_len = bio_labels.shape
    valid_labels = bio_labels.clone()
    for b in range(batch_size):
        for i in range(seq_len):
            curr_label = valid_labels[b, i]
            if curr_label in [1, 3, 5]:  # I-PER, I-LOC, I-ORG
                prev_label = valid_labels[b, i-1] if i > 0 else None
                corresponding_b = curr_label - 1  # B-PER, B-LOC, B-ORG
                if prev_label not in [corresponding_b, curr_label]:
                    valid_labels[b, i] = 6  # O
    return valid_labels

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_encoder=None, vt_encoder=None, vt_hidden_size=0):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.label_encoder = label_encoder
        self.vt_encoder = vt_encoder

        # DDIM Scheduler
        self.noise_scheduler = DDIMScheduler(
            timesteps=self.args.train_steps, 
            device=self.args.device,
            schedule_type="cosine"
        )
        self.noise_scheduler.set_noise_scale(getattr(self.args, 'noise_scale', 0.5))

        # Learnable time embedding
        self.time_embed = nn.Embedding(self.args.train_steps, self.args.embed_dim)
        nn.init.normal_(self.time_embed.weight, mean=0, std=0.02)  # Small random initialization

        self.label_norm = nn.LayerNorm(self.args.label_hidden_dim)

        # Self-attention with relative positional embeddings
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
            ) for _ in range(3)
        ])
        self.label_self_attn_dropout = nn.Dropout(0.1)

        self.vt_proj = nn.Linear(vt_hidden_size, self.args.embed_dim)

        # FiLM for conditioning context on t
        self.context_film = FiLM(self.args.embed_dim)

        # Cross-attention (labels attend on context)
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
            ) for _ in range(2)
        ])
        self.cross_attn_dropout = nn.Dropout(0.2)

        # FiLM for re-conditioning on t
        self.output_film = FiLM(self.args.embed_dim)

        # Output layers
        self.embedding_pred = nn.Sequential(
            nn.Linear(self.args.embed_dim, self.args.embed_dim),
            nn.GELU()
        )
        self.output_dropout = nn.Dropout(0.3)

        # Use GNN decoder for label reconstruction
        self.decoder = self.label_encoder.decoder

    def get_label_embedding(self, labels, attention_mask):
        """Convert label indices to embeddings with GNN and self-attention."""
        assert labels is not None, "labels required"
        assert attention_mask is not None, "attention_mask required"
        labels = labels.to(self.args.device)
        if labels.max() >= self.num_labels:
            logger.warning(f"Label indices out of range: max={labels.max().item()}, num_labels={self.num_labels}")
        
        _, sequence_embeddings, _ = self.label_encoder(edge_index_dict=None, label_indices=labels)
        label_features = sequence_embeddings  # No label_proj, as dimensions match
        label_features = self.label_norm(label_features / torch.norm(label_features, dim=-1, keepdim=True).clamp(min=1e-5))
        label_mask = attention_mask.bool()
        
        for self_attn in self.label_self_attn:
            label_features = self_attn(
                query=label_features, key=label_features, value=label_features, 
                mask=label_mask
            )
            label_features = self.label_self_attn_dropout(label_features)
        
        return label_features, sequence_embeddings

    def get_context_embedding(self, input_ids, attention_mask, token_type_ids=None, 
                             images=None, aux_imgs=None, rcnn_imgs=None):
        """Generate embeddings for visual-textual input."""
        assert input_ids is not None, "input_ids required"
        assert attention_mask is not None, "attention_mask required"
        input_ids = input_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device)
        token_type_ids = token_type_ids.to(self.args.device) if token_type_ids is not None else None
        images = images.to(self.args.device) if images is not None else None
        aux_imgs = aux_imgs.to(self.args.device) if aux_imgs is not None else None
        rcnn_imgs = rcnn_imgs.to(self.args.device) if rcnn_imgs is not None else None
        
        if self.args.ner_model_name == "hvpnet":
            assert images is not None or not self.args.use_prompt, "images required for hvpnet when use_prompt=True"
            assert aux_imgs is not None or not self.args.use_prompt, "aux_imgs required for hvpnet when use_prompt=True"
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.args.ner_model_name == "mkgformer":
            assert rcnn_imgs is not None or not self.args.use_prompt, f"rcnn_imgs required for mkgformer when use_prompt=True"
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
        assert vt_features.shape == (input_ids.size(0), self.args.max_seq_len, self.args.embed_dim)
        return vt_features

    def corrupt(self, t, labels, attention_mask):
        """Corrupt labels with diffusion noise."""
        label_features, clean_embeddings = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, _ = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        return corrupt_label_embeddings, clean_embeddings

    def denoise(self, corrupt_label_embeddings, t, input_ids, attention_mask, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Denoise corrupted embeddings, conditioned on context and t."""
        t = t.to(self.args.device)
        time_features = self.time_embed(t).unsqueeze(1)  # [bsz, 1, embed_dim]
        
        vt_features = self.get_context_embedding(input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs)
        vt_features = self.context_film(vt_features, time_features)
        
        label_features = corrupt_label_embeddings
        attn_mask = attention_mask.bool()
        for attn in self.label_vt_attn:
            label_features = attn(
                query=label_features, key=vt_features, value=vt_features, 
                mask=attn_mask
            )
            label_features = self.cross_attn_dropout(label_features)
        
        label_features = self.output_film(label_features, time_features)
        pred_embeddings = self.output_dropout(self.embedding_pred(label_features))
        
        return pred_embeddings

    def forward(self, labels, input_ids, attention_mask, token_type_ids=None, 
                images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = input_ids.size(0)
        assert attention_mask.max() <= 1 and attention_mask.min() >= 0, "Invalid attention_mask"
        assert labels is not None, "labels required"

        t_random = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        self.t_random = t_random
        corrupt_label_embeddings, clean_embeddings = self.corrupt(t_random, labels, attention_mask)
        pred_embeddings = self.denoise(
            corrupt_label_embeddings, t_random, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs
        )

        # Debug and loss computation
        valid_mask = attention_mask.bool()  # Include all labels, use attention_mask for padding
        valid_mask_1d = valid_mask.view(-1).bool()
        valid_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, self.args.embed_dim)

        # Cosine similarity
        valid_pred = pred_embeddings[valid_mask_3d].view(-1, self.args.embed_dim)
        valid_clean = clean_embeddings[valid_mask_3d].view(-1, self.args.embed_dim)
        cos_sim = F.cosine_similarity(valid_pred, valid_clean, dim=-1).mean()

        # Decoder performance on clean embeddings
        clean_logits = self.decoder(clean_embeddings)
        valid_clean_logits = clean_logits.view(-1, self.num_labels)[valid_mask_1d]
        valid_labels = labels.view(-1)[valid_mask_1d]
        clean_ce_loss = F.cross_entropy(valid_clean_logits, valid_labels, reduction='sum') / valid_mask_1d.sum().clamp(min=1)
        clean_pred_labels = torch.argmax(clean_logits, dim=-1)
        clean_accuracy = (clean_pred_labels[valid_mask] == labels[valid_mask]).float().mean()

        # Standard loss
        logits = self.decoder(pred_embeddings)
        valid_logits = logits.view(-1, self.num_labels)[valid_mask_1d]
        mse_loss = F.mse_loss(
            pred_embeddings * valid_mask_3d,
            clean_embeddings * valid_mask_3d,
            reduction='sum'
        ) / valid_mask_3d.sum().clamp(min=1)
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='sum') / valid_mask_1d.sum().clamp(min=1)

        # Combine losses
        k = getattr(self.args, 'ce_decay_k', 5.0)
        ce_weight = self.args.ce_weight * torch.exp(-k * (self.t_random / self.args.train_steps)).mean()
        loss = mse_loss + ce_weight * ce_loss

        # Logging
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss: mse_loss={mse_loss:.4f}, ce_loss={ce_loss:.4f}, ce_weight={ce_weight:.4f}")
            raise ValueError("Loss is NaN or Inf")
        logger.info(f"Step: Cosine similarity = {cos_sim:.4f}, Clean CE loss = {clean_ce_loss:.4f}, Accuracy = {clean_accuracy:.4f}, "
                    f"mse_loss = {mse_loss:.4f}, ce_loss = {ce_loss:.4f}, ce_weight = {ce_weight:.4f}")

        self.mse_loss = mse_loss
        self.ce_loss = ce_loss
        return loss, logits

    def reverse_diffusion(self, input_ids, attention_mask, token_type_ids=None, 
                         images=None, aux_imgs=None, rcnn_imgs=None, steps=None):
        batch_size, seq_len = input_ids.shape
        steps = steps or getattr(self.args, 'reverse_steps', 50)
        label_embeddings = torch.randn(batch_size, seq_len, self.args.embed_dim, device=self.args.device)
        
        step_size = self.args.train_steps // steps
        timesteps = list(reversed(range(0, self.args.train_steps, step_size)))[:steps]
        
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            pred_embeddings = self.denoise(
                label_embeddings, t_tensor, input_ids, attention_mask, 
                token_type_ids, images, aux_imgs, rcnn_imgs
            )
            eta = getattr(self.args, 'ddim_eta', 0.0)
            label_embeddings = self.noise_scheduler.step(pred_embeddings, t_tensor, label_embeddings, eta=eta)
        
        pred_embeddings = self.denoise(
            label_embeddings, torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs
        )
        
        logits = self.decoder(pred_embeddings)
        pred_labels = torch.argmax(logits, dim=-1)
        if getattr(self.args, 'post_process', True):
            pred_labels = post_process_bio_labels(pred_labels, self.num_labels)
        logger.debug(f"Predicted labels: {pred_labels[0].tolist()}")
        return pred_labels