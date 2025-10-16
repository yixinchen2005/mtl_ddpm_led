import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.attention import UnifiedAttention
import logging
import math

logger = logging.getLogger(__name__)

class NoiseScheduler:
    def __init__(self, device="cpu", schedule_type="cosine"):
        self.device = device
        if schedule_type == "linear":
            self.diffusion_schedule = self.linear_diffusion_schedule
        elif schedule_type == "cosine":
            self.diffusion_schedule = self.cosine_diffusion_schedule
        elif schedule_type == "offset_cosine":
            self.diffusion_schedule = self.offset_cosine_diffusion_schedule
        elif schedule_type == "cosine_2":
            self.diffusion_schedule = self.cosine_diffusion_schedule_2
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
    
    def linear_diffusion_schedule(self, diffusion_times, beta_start=1e-4, beta_end=0.02):
        diffusion_times = diffusion_times.to(device=self.device, dtype=torch.float32)
        integral = beta_start * diffusion_times + 0.5 * (beta_end - beta_start) * diffusion_times**2
        alpha_bars = torch.exp(-integral)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1 - alpha_bars)
        return noise_rates, signal_rates
    
    def cosine_diffusion_schedule(self, diffusion_times):
        diffusion_times = diffusion_times.to(device=self.device, dtype=torch.float32)
        signal_rates = torch.cos(diffusion_times * torch.pi / 2)
        noise_rates = torch.sin(diffusion_times * torch.pi / 2)
        return noise_rates, signal_rates
    
    def offset_cosine_diffusion_schedule(self, diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
        diffusion_times = diffusion_times.to(device=self.device, dtype=torch.float32)
        angle_start = torch.acos(min_signal_rate)
        angle_end = torch.acos(max_signal_rate)
        diffusion_angles = angle_start + diffusion_times * (angle_end - angle_start)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates
    
    def cosine_diffusion_schedule_2(self, diffusion_times, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        diffusion_times = diffusion_times.to(device=self.device, dtype=torch.float32)
        alpha_bars = (torch.cos(((diffusion_times + s) / (1 + s)) * math.pi * 0.5) ** 2) / \
           (math.cos((s / (1 + s)) * math.pi * 0.5) ** 2)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1. - alpha_bars)
        return noise_rates, signal_rates
    
    def add_noise(self, embedding, diffusion_times):
        """
        embedding: [bsz, seq_len, dim]
        diffusion_times: LongTensor shape [bsz] with values in [0, 1]
        returns noisy_embedding, eps (noise)
        """
        assert diffusion_times.max().item() <= 1.0 and diffusion_times.min().item() >= 0.0, f"Invalid diffusion times: {diffusion_times}"
        noise = torch.randn_like(embedding)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times=diffusion_times)
        noise_rates = noise_rates.view(-1, 1, 1)
        signal_rates = signal_rates.view(-1, 1, 1)
        noisy_embedding = signal_rates * embedding + noise_rates * noise
        return noisy_embedding, noise

class FiLM(nn.Module):
    def __init__(self, dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
    
    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * gamma + beta
    
class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding layer (like in diffusion models or transformers).
    
    Args:
        noise_embedding_size (int): output embedding size (must be even).
    """
    def __init__(self, noise_embedding_size):
        super().__init__()
        if noise_embedding_size % 2 != 0:
            raise ValueError("noise_embedding_size must be even")

        self.noise_embedding_size = noise_embedding_size

        # Precompute frequencies (fixed, not learnable)
        frequencies = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(1.0)),
                torch.log(torch.tensor(1000.0)),
                steps=noise_embedding_size // 2,
            )
        )
        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies, persistent=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size,)
        Returns:
            Tensor of shape (batch_size, noise_embedding_size)
        """
        x = x.unsqueeze(-1)
        embeddings = torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)],
            dim=-1
        )
        return embeddings
    
class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_encoder=None, vt_encoder=None, label_embedding_normalizer=None, vt_hidden_size=0):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.label_encoder = label_encoder
        self.vt_encoder = vt_encoder
        self.label_embedding_normalizer = label_embedding_normalizer

        # DDIM Scheduler
        self.noise_scheduler = NoiseScheduler(
            device=self.args.device,
            schedule_type="cosine_2"
        )

        self.rate_embed = SinusoidalEmbedding(self.args.embed_dim)

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

    def get_vt_embedding(self, input_ids, attention_mask, token_type_ids=None, 
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
            vt_embedding = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
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
            vt_embedding = out.last_hidden_state
        vt_embedding = self.vt_proj(vt_embedding)
        assert vt_embedding.shape == (input_ids.size(0), self.args.max_seq_len, self.args.embed_dim)
        return vt_embedding
    
    def corrupt(self, diffusion_times, labels):
        """Corrupt labels with diffusion noise."""
        assert labels is not None, "labels required"
        labels = labels.to(self.args.device)
        if labels.max() >= self.num_labels:
            logger.warning(f"Label indices out of range: max={labels.max().item()}, num_labels={self.num_labels}")
        
        _, label_embedding, _ = self.label_encoder(edge_index_dict=None, label_indices=labels)
        label_embedding = self.label_embedding_normalizer(label_embedding) # To guarantee the corrupted embedding approximate unit Gaussian noise
        noisy_label_embedding, noise = self.noise_scheduler.add_noise(label_embedding, diffusion_times)
        return noisy_label_embedding, label_embedding, noise
    
    def denoise(self, noisy_label_embeddings, diffusion_times, input_ids, attention_mask, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        """Denoise corrupted embeddings, conditioned on context and noise rate."""
        noise_rates, signal_rates = self.noise_scheduler.diffusion_schedule(diffusion_times=diffusion_times)
        # The network is designed to predict the clean label embeddings, so provide the signal rate embeddings #
        signal_rate_embeddings = self.rate_embed(signal_rates**2)
        signal_rate_embeddings = signal_rate_embeddings.unsqueeze(1) # [bsz, 1, embed_dim]
        
        vt_embeddings = self.get_vt_embedding(input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs)
        vt_embeddings = self.context_film(vt_embeddings, signal_rate_embeddings)
        
        label_embeddings = noisy_label_embeddings.clone()
        attn_mask = attention_mask.bool()
        
        # Self-attention #
        for self_attn in self.label_self_attn:
            label_embeddings = self_attn(
                query=label_embeddings, key=label_embeddings, value=label_embeddings, 
                mask=attn_mask
            )
            label_embeddings = self.label_self_attn_dropout(label_embeddings)

        # Cross-attention #
        for attn in self.label_vt_attn:
            label_embeddings = attn(
                query=label_embeddings, key=vt_embeddings, value=vt_embeddings, 
                mask=attn_mask
            )
            label_embeddings = self.cross_attn_dropout(label_embeddings)
        
        label_embeddings = self.output_film(label_embeddings, signal_rate_embeddings)
        pred_label_embeddings = self.output_dropout(self.embedding_pred(label_embeddings))

        pred_noises = (noisy_label_embeddings - signal_rates.view(-1, 1, 1) * pred_label_embeddings) / noise_rates.view(-1, 1, 1)
        
        return pred_noises, pred_label_embeddings
    
    def forward(self, labels, input_ids, attention_mask, token_type_ids=None, 
                images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = input_ids.size(0)
        assert attention_mask.max() <= 1 and attention_mask.min() >= 0, "Invalid attention_mask"
        assert labels is not None, "labels required"

        diffusion_times = torch.rand((bsz,), device=self.args.device, dtype=torch.float32)
        noisy_label_embeddings, label_embeddings, _ = self.corrupt(diffusion_times, labels)
        _, pred_label_embeddings = self.denoise(
            noisy_label_embeddings, diffusion_times, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs
        )

        # Debug and loss computation
        valid_mask = attention_mask.bool()  # Include all labels, use attention_mask for padding
        valid_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, self.args.embed_dim)

        # Cosine similarity
        valid_pred = pred_label_embeddings[valid_mask_3d].view(-1, self.args.embed_dim)
        valid_clean = label_embeddings[valid_mask_3d].view(-1, self.args.embed_dim)
        cos_sim = F.cosine_similarity(valid_pred, valid_clean, dim=-1).mean()

        # Decoder performance on clean embeddings
        orig_label_embeddings = self.label_embedding_normalizer.denormalize(label_embeddings)
        clean_logits = self.decoder(orig_label_embeddings)
        clean_pred_labels = torch.argmax(clean_logits, dim=-1)
        clean_accuracy = (clean_pred_labels[valid_mask] == labels[valid_mask]).float().mean()

        # Standard loss
        loss = F.mse_loss(
            pred_label_embeddings * valid_mask_3d,
            label_embeddings * valid_mask_3d,
            reduction='sum'
        ) / valid_mask_3d.sum().clamp(min=1)
        orig_pred_label_embeddings = self.label_embedding_normalizer.denormalize(pred_label_embeddings)
        logits = self.decoder(orig_pred_label_embeddings)

        # Logging
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss: loss={loss:.4f}")
            raise ValueError("Loss is NaN or Inf")
        logger.info(f"Step: Cosine similarity = {cos_sim:.4f}, Accuracy = {clean_accuracy:.4f}, "f"loss = {loss:.4f}")

        return loss, logits
    
    def reverse_diffusion(self, initial_noise, reverse_steps, input_ids, attention_mask, token_type_ids=None,
                      images=None, aux_imgs=None, rcnn_imgs=None, eta=0.0):
        step_size = 1.0 / reverse_steps
        noise = torch.randn(self.args.batch_size, self.args.max_seq_len, self.args.embed_dim, device=self.args.device)
        current_embeddings = initial_noise
        for step in range(reverse_steps):
            diffusion_times = torch.ones((self.args.batch_size,)) - step * step_size
            noise_rates, signal_rates = self.noise_scheduler.diffusion_schedule(diffusion_times=diffusion_times)
            pred_noises, pred_label_embeddings = self.denoise(current_embeddings, diffusion_times, input_ids, attention_mask, \
                token_type_ids, images, aux_imgs, rcnn_imgs)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.noise_scheduler.diffusion_schedule(next_diffusion_times)
            noise_rates = noise_rates.view(-1, 1, 1)
            signal_rates = signal_rates.view(-1, 1, 1)
            next_noise_rates = next_noise_rates.view(-1, 1, 1)
            next_signal_rates = next_signal_rates.view(-1, 1, 1)
            current_embeddings = next_signal_rates * pred_label_embeddings + \
                next_noise_rates * torch.sqrt(1 - eta**2 * (next_signal_rates**2 - signal_rates**2) / (next_signal_rates**2 * noise_rates**2)) * pred_noises + \
                eta * torch.sqrt(((next_signal_rates**2 - signal_rates**2) * next_noise_rates**2) / (next_signal_rates**2 * noise_rates**2)) * noise
        return pred_label_embeddings
    
    def generate(self, input_ids, attention_mask, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn(self.args.batch_size, self.args.max_seq_len, self.args.embed_dim, device=self.args.device)
        generated_label_embedding = self.reverse_diffusion(initial_noise, self.args.reverse_steps, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, self.args.eta)
        generated_label_embedding = self.label_embedding_normalizer.denormalize(generated_label_embedding)
        generated_label_logits = self.decoder(generated_label_embedding)
        generated_labels = torch.argmax(generated_label_logits, dim=-1)
        return generated_labels