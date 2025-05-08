import torch
import os
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
from .char_lstm import CharLSTM
from .bert_model import HMNeTNERModel
from .unimo_model import UnimoCRFModel
from utils.attention import MultiAttn, PositionalEncoding

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        assert 0 < beta_start < beta_end < 1, "Invalid beta range"
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t, attention_mask=None):
        assert t.max() < self.timesteps and t.min() >= 0, "Invalid timestep"
        noise = torch.randn_like(x)
        signal_rate_t = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        noise_rate_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        noisy_x = signal_rate_t * x + noise_rate_t * noise
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            noisy_x = mask * noisy_x + (1 - mask) * x
        return noisy_x, noise

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels=0, label_embedding_table=None, clstm_path=None, ner_model_name="hvpnet"):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.ner_model_name = ner_model_name
        self.pseudo_label_f1_threshold = 0.75
        self.use_pseudo_labels = False
        self.current_ner_f1 = 0.0
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
        self.noise_scheduler = NoiseScheduler(timesteps=self.args.train_steps, device=self.args.device)
        # Label Encoder
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
        # CharLSTM
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
        # Visual-Textual Clue Encoder
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
        # Cross Attention (1st Block)
        self.label_vt_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,  # 256
            key_dim=vt_hidden_size,               # 768
            value_dim=vt_hidden_size,             # 768
            emb_dim=self.args.label_hidden_dim,   # 256
            num_heads=4, 
            dropout_rate=0.4
        )
        self.label_char_attn = MultiAttn(
            query_dim=self.args.label_hidden_dim,  # 256
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=self.args.label_hidden_dim,   # 256
            num_heads=4,
            dropout_rate=0.4
        )
        self.char_vt_attn = MultiAttn(
            query_dim=self.args.char_hidden_dim,  # 512
            key_dim=vt_hidden_size,              # 768
            value_dim=vt_hidden_size,            # 768
            emb_dim=self.args.char_hidden_dim,   # 512
            num_heads=4, 
            dropout_rate=0.4
        )
        self.char_label_attn = MultiAttn(
            query_dim=self.args.char_hidden_dim,   # 512
            key_dim=self.args.label_hidden_dim,   # 256
            value_dim=self.args.label_hidden_dim, # 256
            emb_dim=self.args.char_hidden_dim,    # 512
            num_heads=4, 
            dropout_rate=0.4
        )
        self.vt_label_attn = MultiAttn(
            query_dim=vt_hidden_size,             # 768
            key_dim=self.args.label_hidden_dim,   # 256
            value_dim=self.args.label_hidden_dim, # 256
            emb_dim=vt_hidden_size,              # 768
            num_heads=4, 
            dropout_rate=0.4
        )
        self.vt_char_attn = MultiAttn(
            query_dim=vt_hidden_size,             # 768
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=vt_hidden_size,              # 768
            num_heads=4,
            dropout_rate=0.4
        )
        # Cross Attention (2nd Block)
        self.refine_label_char = MultiAttn(
            query_dim=self.args.label_hidden_dim,  # 256
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=self.args.label_hidden_dim,   # 256
            num_heads=4, 
            dropout_rate=0.4
        )
        self.refine_vt_char_label = MultiAttn(
            query_dim=vt_hidden_size,             # 768
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=vt_hidden_size,              # 768
            num_heads=4, 
            dropout_rate=0.4
        )
        self.refine_char_label = MultiAttn(
            query_dim=self.args.char_hidden_dim,   # 512
            key_dim=self.args.label_hidden_dim,   # 256
            value_dim=self.args.label_hidden_dim, # 256
            emb_dim=self.args.char_hidden_dim,    # 512
            num_heads=4, 
            dropout_rate=0.4
        )
        self.refine_label_char_context = MultiAttn(
            query_dim=self.args.label_hidden_dim,  # 256
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=self.args.label_hidden_dim,   # 256
            num_heads=4,
            dropout_rate=0.4
        )
        self.refine_char_vt = MultiAttn(
            query_dim=self.args.char_hidden_dim,  # 512
            key_dim=vt_hidden_size,              # 768
            value_dim=vt_hidden_size,            # 768
            emb_dim=self.args.char_hidden_dim,   # 512
            num_heads=4, 
            dropout_rate=0.4
        )
        self.refine_vt_char = MultiAttn(
            query_dim=vt_hidden_size,             # 768
            key_dim=self.args.char_hidden_dim,    # 512
            value_dim=self.args.char_hidden_dim,  # 512
            emb_dim=vt_hidden_size,              # 768
            num_heads=4,
            dropout_rate=0.4
        )
        # Normalization
        self.norm_char_label = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_label = nn.LayerNorm(self.vt_hidden_size)
        self.norm_label_vt = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_char = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_char_vt = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_char = nn.LayerNorm(self.vt_hidden_size)
        self.norm_char_label_2 = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_label_2 = nn.LayerNorm(self.vt_hidden_size)
        self.norm_label_vt_2 = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_label_char_2 = nn.LayerNorm(self.args.label_hidden_dim)
        self.norm_char_vt_2 = nn.LayerNorm(self.args.char_hidden_dim)
        self.norm_vt_char_2 = nn.LayerNorm(self.vt_hidden_size)
        # Output Layers
        self.fc = nn.Sequential(
            nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, num_labels),
            nn.LayerNorm(num_labels)
        )
        self.noise_pred = nn.Linear(self.vt_hidden_size + self.args.label_hidden_dim, self.args.label_hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, labels=None, targets_new=None, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, mode='pretrain', epoch=0):
        self.current_epoch = epoch
        bsz = input_ids.size(0) if input_ids is not None else labels.size(0)
        assert attention_mask.max() <= 1 and attention_mask.min() >= 0, "Invalid attention_mask"

        if self.ner_model_name == "hvpnet":
            ner_loss, ner_logits, ner_probs = self.ner_model(input_ids, attention_mask, token_type_ids, 
                                                            labels, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            ner_loss, ner_logits, ner_probs = self.ner_model(input_ids, attention_mask, token_type_ids, 
                                                            labels, images, aux_imgs, rcnn_imgs)
        else:
            raise ValueError("Invalid ner_model_name")
        ner_loss = ner_loss if ner_loss is not None else torch.tensor(0.0, device=self.args.device)

        if self.ner_model_name == "hvpnet":
            sequence_output = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            sequence_output = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs).last_hidden_state
        else:
            raise ValueError("Invalid ner_model_name")
        ner_emissions = self.ner_model.fc(sequence_output)

        if labels is None:
            return None, None, ner_logits
        
        t = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        corrupt_label_embeddings, noise = self.corrupt(t, labels, attention_mask)
        recon_emissions, predicted_noise = self.denoise(
            corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs, targets_new=targets_new if mode in ['finetune_forward', 'finetune_backward'] else None
        )
        
        mse_loss = F.mse_loss(predicted_noise, noise)
        if mode in ['finetune_forward', 'finetune_backward'] and targets_new is not None:
            error_mask = (targets_new != labels).float()
            denoise_crf_loss = -self.crf(recon_emissions, targets_new, mask=attention_mask.bool(), reduction='none')
            denoise_crf_loss = (denoise_crf_loss * (1.0 + 4.0 * error_mask)).mean()
            loss = 1.0 * denoise_crf_loss + 0.3 * mse_loss
        else:
            pseudo_labels = ner_emissions.argmax(dim=-1).detach() if self.use_pseudo_labels else labels
            denoise_crf_loss = -self.crf(recon_emissions, pseudo_labels, mask=attention_mask.bool(), reduction='mean')
            recon_probs = F.softmax(recon_emissions, dim=-1)
            kl_loss = F.kl_div(recon_probs.log(), ner_probs, reduction='batchmean') if self.use_pseudo_labels else 0.0
            loss = 0.8 * ner_loss + 0.5 * mse_loss + 0.1 * denoise_crf_loss + 0.1 * kl_loss
        
        self.ner_loss = ner_loss
        self.mse_loss = mse_loss
        self.denoise_crf_loss = denoise_crf_loss
        self.kl_loss = torch.tensor(0.0, device=self.args.device) if mode in ['finetune_forward', 'finetune_backward'] else kl_loss
        
        return loss, recon_emissions, ner_logits
    
    def reverse_diffusion(self, char_input_ids, input_ids, attention_mask, token_type_ids, 
                         images, aux_imgs, rcnn_imgs, steps=100):
        batch_size, seq_len = input_ids.shape
        label_embeddings = torch.randn(batch_size, seq_len, self.args.label_hidden_dim, device=self.args.device)
        
        for t in range(steps - 1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=self.args.device, dtype=torch.long)
            _, predicted_noise, _ = self.denoise(
                label_embeddings, t_tensor, char_input_ids, input_ids, attention_mask,
                token_type_ids, images, aux_imgs, rcnn_imgs, return_features=True
            )
            alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1, 1)
            alpha_t = self.noise_scheduler.alpha[t].view(-1, 1, 1)
            sigma_t = torch.sqrt(1 - alpha_bar_t) * torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_bar_t)
            coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            label_embeddings = (label_embeddings - coeff * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                z = torch.randn_like(label_embeddings)
                label_embeddings += sigma_t * z
        
        recon_emissions, _, _ = self.denoise(
            label_embeddings, torch.zeros(batch_size, device=self.args.device, dtype=torch.long),
            char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs,
            return_features=True
        )
        diffusion_logits = recon_emissions.argmax(dim=-1)
        return diffusion_logits
    
    def corrupt(self, t, labels=None, attention_mask=None):
        label_features = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        return corrupt_label_embeddings, noise
    
    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, 
                token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, targets_new=None, return_features=False):
        t = t.float().view(-1, 1)
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)
        char_features, vt_features = self.get_context_embedding(char_input_ids, input_ids, attention_mask, 
                                                              token_type_ids, images, aux_imgs, rcnn_imgs)
        attn_mask = 1 - attention_mask if attention_mask is not None else None
        # Add time features to condition denoising on timestep t
        corrupt_label_embeddings = corrupt_label_embeddings + time_features

        # Process targets_new for fine-tuning
        new_label_features = None
        if targets_new is not None:
            new_label_features = self.get_label_embedding(targets_new, attention_mask)

        # First Cross-Attention Block
        char_label_features = self.char_label_attn(
            query=char_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        )
        vt_label_features = self.vt_label_attn(
            query=vt_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attn_mask
        )
        label_vt_features = self.label_vt_attn(
            query=corrupt_label_embeddings, key=vt_features, value=vt_features, mask=attn_mask
        )
        label_char_features = self.label_char_attn(
            query=corrupt_label_embeddings, key=char_features, value=char_features, mask=attn_mask
        )
        char_vt_features = self.char_vt_attn(
            query=char_features, key=vt_features, value=vt_features, mask=attn_mask
        )
        vt_char_features = self.vt_char_attn(
            query=vt_features, key=char_features, value=char_features, mask=attn_mask
        )
        if new_label_features is not None:
            label_vt_features = label_vt_features + self.norm_label_vt(new_label_features)
            label_char_features = label_char_features + self.norm_label_char(new_label_features)

        # Normalize and Residual
        char_label_features = self.norm_char_label(char_label_features + char_features)
        vt_label_features = self.norm_vt_label(vt_label_features + vt_features)
        label_vt_features = self.norm_label_vt(label_vt_features + corrupt_label_embeddings)
        label_char_features = self.norm_label_char(label_char_features + corrupt_label_embeddings)
        char_vt_features = self.norm_char_vt(char_vt_features + char_features)
        vt_char_features = self.norm_vt_char(vt_char_features + vt_features)

        # Second Cross-Attention Block
        label_vt_features_2 = self.refine_label_char(
            query=label_vt_features, key=char_vt_features, value=char_vt_features, mask=attn_mask
        )
        vt_label_features_2 = self.refine_vt_char_label(
            query=vt_features, key=char_label_features, value=char_label_features, mask=attn_mask
        )
        char_vt_features_2 = self.refine_char_label(
            query=char_label_features, key=label_vt_features, value=label_vt_features, mask=attn_mask
        )
        label_char_features_2 = self.refine_label_char_context(
            query=label_char_features, key=char_vt_features, value=char_vt_features, mask=attn_mask
        )
        char_label_features_2 = self.refine_char_vt(
            query=char_label_features, key=vt_label_features, value=vt_label_features, mask=attn_mask
        )
        vt_char_features_2 = self.refine_vt_char(
            query=vt_features, key=char_vt_features, value=char_vt_features, mask=attn_mask
        )

        if new_label_features is not None:
            label_vt_features_2 = label_vt_features_2 + self.norm_label_vt_2(new_label_features)
            label_char_features_2 = label_char_features_2 + self.norm_label_char_2(new_label_features)

        # Normalize
        char_label_features_2 = self.norm_char_label_2(char_label_features_2)
        vt_label_features_2 = self.norm_vt_label_2(vt_label_features_2)
        label_vt_features_2 = self.norm_label_vt_2(label_vt_features_2)
        label_char_features_2 = self.norm_label_char_2(label_char_features_2)
        char_vt_features_2 = self.norm_char_vt_2(char_vt_features_2)
        vt_char_features_2 = self.norm_vt_char_2(vt_char_features_2)

        # Feature Combination
        label_features_comb = (label_vt_features_2 + label_char_features_2) / 2
        vt_features_comb = (vt_label_features_2 + vt_char_features_2) / 2
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1)
        
        features = self.dropout(features)
        recon_emissions = self.fc(features)
        predicted_noise = self.noise_pred(features)

        if return_features:
            return recon_emissions, predicted_noise, features
        return recon_emissions, predicted_noise
    
    def get_label_embedding(self, labels, attention_mask=None, is_targets_new=False):
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
                             images=None, aux_imgs=None, rcnn_imgs=None):
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
            attn_mask = 1 - attention_mask if attention_mask is not None else None
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
    
    def update_pseudo_label_state(self, ner_f1):
        self.current_ner_f1 = ner_f1
        if ner_f1 >= self.pseudo_label_f1_threshold:
            self.use_pseudo_labels = True