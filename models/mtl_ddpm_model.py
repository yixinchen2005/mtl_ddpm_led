import torch, os
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
from copy import deepcopy
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
            mask = attention_mask.unsqueeze(-1).float()  # (bsz, 128, 1)
            noisy_x = mask * noisy_x + (1 - mask) * x  # Preserve padded tokens
        return noisy_x, noise

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels, label_embedding_table, clstm_path, ner_model_name="hvpnet"):
        super().__init__()
        # Configurations #
        self.args = args
        self.num_labels = num_labels
        self.label_embedding_table = label_embedding_table
        self.ner_model_name = ner_model_name
        # Noise Scheduler #
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
        self.noise_scheduler = NoiseScheduler(timesteps=self.args.train_steps, device=self.args.device)
        # CharLSTM (from prior)
        char2int_dict, int2char_dict = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        self.char_lstm = CharLSTM(char2int_dict, int2char_dict, n_hidden=args.char_hidden_dim, 
                                 n_layers=2, bidirectional=True, drop_prob=0.3)
        self.char_lstm.load_state_dict(torch.load(os.path.join(clstm_path, "char_lstm.pth")))
        self.char_lstm_mlp = nn.Linear(2 * self.char_lstm.n_layers * args.char_hidden_dim, args.char_hidden_dim)
        self.char_pos_encoder = PositionalEncoding(args.char_hidden_dim, args.max_seq_len)
        self.char_self_attn = MultiAttn(query_dim=args.char_hidden_dim, key_dim=args.char_hidden_dim, 
                                       value_dim=args.char_hidden_dim, emb_dim=args.char_hidden_dim, 
                                       num_heads=4, dropout_rate=0.3)

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

        # Cross Attention (1st Block)
        self.label_vt_attn = MultiAttn(query_dim=args.label_hidden_dim, key_dim=vt_hidden_size, 
                                      value_dim=vt_hidden_size, emb_dim=vt_hidden_size, num_heads=4, dropout_rate=0.4)
        self.char_vt_attn = MultiAttn(query_dim=args.char_hidden_dim, key_dim=vt_hidden_size, 
                                     value_dim=vt_hidden_size, emb_dim=vt_hidden_size, num_heads=4, dropout_rate=0.4)
        self.vt_label_attn = MultiAttn(query_dim=vt_hidden_size, key_dim=args.label_hidden_dim, 
                                      value_dim=args.label_hidden_dim, emb_dim=args.label_hidden_dim, 
                                      num_heads=4, dropout_rate=0.4)
        self.char_label_attn = MultiAttn(query_dim=args.char_hidden_dim, key_dim=args.label_hidden_dim, 
                                       value_dim=args.label_hidden_dim, emb_dim=args.label_hidden_dim, 
                                       num_heads=4, dropout_rate=0.4)

        # Cross Attention (2nd Block)
        self.label_vt_attn_2 = MultiAttn(query_dim=args.label_hidden_dim, key_dim=vt_hidden_size, 
                                        value_dim=vt_hidden_size, emb_dim=vt_hidden_size, num_heads=4, dropout_rate=0.4)
        self.char_vt_attn_2 = MultiAttn(query_dim=args.char_hidden_dim, key_dim=vt_hidden_size, 
                                       value_dim=vt_hidden_size, emb_dim=vt_hidden_size, num_heads=4, dropout_rate=0.4)
        self.vt_label_attn_2 = MultiAttn(query_dim=vt_hidden_size, key_dim=args.label_hidden_dim, 
                                        value_dim=args.label_hidden_dim, emb_dim=args.label_hidden_dim, 
                                        num_heads=4, dropout_rate=0.4)
        self.char_label_attn_2 = MultiAttn(query_dim=args.char_hidden_dim, key_dim=args.label_hidden_dim, 
                                         value_dim=args.label_hidden_dim, emb_dim=args.label_hidden_dim, 
                                         num_heads=4, dropout_rate=0.4)

        # Normalization
        self.norm_char_label = nn.LayerNorm(args.char_hidden_dim)
        self.norm_vt_label = nn.LayerNorm(args.label_hidden_dim)
        self.norm_label_vt = nn.LayerNorm(vt_hidden_size)
        self.norm_char_vt = nn.LayerNorm(vt_hidden_size)
        self.norm_char_label_2 = nn.LayerNorm(args.char_hidden_dim)
        self.norm_vt_label_2 = nn.LayerNorm(args.label_hidden_dim)
        self.norm_label_vt_2 = nn.LayerNorm(vt_hidden_size)
        self.norm_char_vt_2 = nn.LayerNorm(vt_hidden_size)

        # Output Layers
        self.fc = nn.Sequential(
            nn.Linear(vt_hidden_size + args.label_hidden_dim, num_labels),
            nn.LayerNorm(num_labels)
        )
        self.noise_pred = nn.Linear(vt_hidden_size + self.args.label_hidden_dim, self.args.label_hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, labels, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, 
                images=None, aux_imgs=None, rcnn_imgs=None, mode='pretrain', epoch=0):
        bsz = labels.size(0)
        assert attention_mask.max() <= 1 and attention_mask.min() >= 0, "Invalid attention_mask"
        self.current_epoch = epoch
        
        # Sample timesteps
        t = torch.randint(0, self.args.train_steps, (bsz,), device=self.args.device)
        
        # Corrupt labels
        corrupt_label_embeddings, noise = self.corrupt(t, labels, attention_mask)
        
        # Denoise
        recon_emissions, predicted_noise, features = self.denoise(
            corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, 
            token_type_ids, images, aux_imgs, rcnn_imgs, return_features=True
        )
        
        # Diffusion loss
        mse_loss = F.mse_loss(predicted_noise, noise)
        
        # CRF loss from ner_model
        if self.ner_model_name == "hvpnet":
            crf_loss, crf_logits, crf_probs = self.ner_model(input_ids, attention_mask, token_type_ids, 
                                                            labels, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            crf_loss, crf_logits, crf_probs = self.ner_model(input_ids, attention_mask, token_type_ids, 
                                                            labels, images, aux_imgs, rcnn_imgs)
        else:
            raise ValueError("Invalid ner_model_name")
        crf_loss = crf_loss if crf_loss is not None else torch.tensor(0.0, device=self.args.device)
        
        # CRF loss from denoise
        pseudo_labels = labels  # Use targets_unk in pre-training
        denoise_crf_loss = -self.crf(recon_emissions, pseudo_labels, mask=attention_mask.bool(), reduction='mean')
        
        # KL divergence (disabled in pre-training)
        recon_probs = F.softmax(recon_emissions, dim=-1)
        kl_loss = F.kl_div(recon_probs.log(), crf_probs, reduction='batchmean') if mode != 'pretrain' else 0.0
        
        # Loss weights with scheduling
        denoise_weight = 0.2 * min(1.0, (self.current_epoch - 4) / 5) if mode == 'pretrain' and self.current_epoch >= 5 else 0.0
        mse_weight = 1.5 if mode == 'pretrain' and self.current_epoch < 5 else 1.0
        
        # Combine losses
        if mode == 'pretrain':
            loss = mse_weight * mse_loss + 0.5 * crf_loss + denoise_weight * denoise_crf_loss
        elif mode == 'finetune_forward':
            loss = denoise_crf_loss + 0.5 * crf_loss + 0.1 * kl_loss
        elif mode == 'finetune_backward':
            loss = denoise_crf_loss + 0.5 * crf_loss + 0.1 * kl_loss
        else:
            raise ValueError("Invalid mode")
        
        return loss, recon_emissions
    
    def corrupt(self, t, labels=None, attention_mask=None):
        label_features = self.get_label_embedding(labels, attention_mask)
        corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t, attention_mask)
        return corrupt_label_embeddings, noise
    
    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, return_features=False):
        t = t.float().view(-1, 1)
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)  # (bsz, 1, 256)
        char_features, vt_features = self.get_context_embedding(char_input_ids, input_ids, attention_mask, 
                                                              token_type_ids, images, aux_imgs, rcnn_imgs)
        attn_mask = 1 - attention_mask if attention_mask is not None else None
        corrupt_label_embeddings = corrupt_label_embeddings + time_features  # (bsz, 128, 256)

        # First Cross-Attention Block
        char_label_features = self.char_label_attn(query=char_features, key=corrupt_label_embeddings, 
                                                 value=corrupt_label_embeddings, mask=attn_mask)  # (bsz, 128, 256)
        vt_label_features = self.vt_label_attn(query=vt_features, key=corrupt_label_embeddings, 
                                              value=corrupt_label_embeddings, mask=attn_mask)  # (bsz, 128, 256)
        label_vt_features = self.label_vt_attn(query=corrupt_label_embeddings, key=vt_features, 
                                              value=vt_features, mask=attn_mask)  # (bsz, 128, 768)
        char_vt_features = self.char_vt_attn(query=char_features, key=vt_features, 
                                           value=vt_features, mask=attn_mask)  # (bsz, 128, 768)

        # Normalize and Residual
        char_label_features = self.norm_char_label(char_label_features + char_features)
        vt_label_features = self.norm_vt_label(vt_label_features + corrupt_label_embeddings)
        label_vt_features = self.norm_label_vt(label_vt_features + vt_features)
        char_vt_features = self.norm_char_vt(char_vt_features + vt_features)

        # Second Cross-Attention Block
        char_label_features_2 = self.char_label_attn_2(query=char_label_features, key=vt_label_features, 
                                                     value=vt_label_features, mask=attn_mask)  # (bsz, 128, 256)
        vt_label_features_2 = self.vt_label_attn_2(query=vt_features, key=char_label_features, 
                                                  value=char_label_features, mask=attn_mask)  # (bsz, 128, 256)
        label_vt_features_2 = self.label_vt_attn_2(query=vt_label_features, key=char_vt_features, 
                                                  value=char_vt_features, mask=attn_mask)  # (bsz, 128, 768)
        char_vt_features_2 = self.char_vt_attn_2(query=char_label_features, key=label_vt_features, 
                                                value=label_vt_features, mask=attn_mask)  # (bsz, 128, 768)

        # Normalize
        char_label_features_2 = self.norm_char_label_2(char_label_features_2)
        vt_label_features_2 = self.norm_vt_label_2(vt_label_features_2)
        label_vt_features_2 = self.norm_label_vt_2(label_vt_features_2)
        char_vt_features_2 = self.norm_char_vt_2(char_vt_features_2)

        # Feature Combination
        label_features_comb = char_label_features_2 + vt_label_features_2  # (bsz, 128, 256)
        vt_features_comb = char_vt_features_2 + label_vt_features_2  # (bsz, 128, 768)
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1)  # (bsz, 128, 1024)
        
        features = self.dropout(features)
        recon_emissions = self.fc(features)  # (bsz, 128, 13)
        predicted_noise = self.noise_pred(features)

        if return_features:
            return recon_emissions, predicted_noise, features
        return recon_emissions, predicted_noise
    
    def get_label_embedding(self, labels, attention_mask=None):
        assert labels is not None, "labels required"
        assert labels.max() < self.label_embedding_table.shape[0], "Label indices out of range"
        
        label_features = self.label_embedding_table[labels]  # (batch_size, seq_len, 768)
        label_features = self.label_mlp(label_features)  # (batch_size, seq_len, label_hidden_dim)
        label_features = self.label_pos_encoder(label_features)  # (batch_size, seq_len, label_hidden_dim)
        
        label_mask = 1 - attention_mask if attention_mask is not None else None
        if attention_mask is not None:
            print(f"label_mask sample: {label_mask[0].tolist()}")
        
        label_features = self.label_self_attn(
            query=label_features,
            key=label_features,
            value=label_features,
            mask=label_mask
        )  # (batch_size, seq_len, label_hidden_dim)
        
        return label_features
    
    def get_context_embedding(self, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = input_ids.size(0)
        
        # Char Features
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

        # Visual-Textual Features
        if self.ner_model_name == "hvpnet":
            vt_features = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                  pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
            vt_features = out.last_hidden_state
        assert vt_features.shape == (bsz, self.args.max_seq_len, 768), "VT output shape mismatch"
        
        return char_features, vt_features
