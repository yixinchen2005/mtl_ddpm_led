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
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)  # Linear schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # Cumulative product

    def add_noise(self, x, t):
        """Corrupts x by adding Gaussian noise at time step t"""
        noise = torch.randn_like(x)
        signal_rate_t = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        noise_rate_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        return signal_rate_t * x + noise_rate_t * noise, noise

class DiffusionModel(nn.Module):
    def __init__(self, args, num_labels, label_embedding_table, clstm_path, ner_model_name="hvpnet"):
        super().__init__()
        # Configurations #
        self.args = args
        self.num_labels = num_labels
        self.label_embedding_table = label_embedding_table
        self.ner_model_name = ner_model_name
        # Time #
        self.time_mlp = nn.Linear(1, self.args.time_hidden_dim)
        # Char LSTM #
        char2int_dict, int2char_dict = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        self.char_lstm = CharLSTM(char2int_dict=char2int_dict, int2char_dict=int2char_dict, n_hidden=self.args.char_hidden_dim, bidirectional=True)
        self.char_lstm.load_state_dict(torch.load(os.path.join(clstm_path, "char_lstm.pth")))
        self.char_lstm_mlp = nn.Linear(4*self.args.char_hidden_dim, self.args.char_hidden_dim)
        # Positional Encoders #
        self.char_pos_encoder = PositionalEncoding(4*self.args.char_hidden_dim, self.args.max_seq_len)
        self.label_pos_encoder = PositionalEncoding(self.args.label_hidden_dim, self.args.max_seq_len)
        # Visual-Textual Clue Encoder #
        if self.ner_model_name == "hvpnet":
            self.ner_model = HMNeTNERModel(self.num_labels, args)
            self.vt_encoder = self.ner_model.core
        elif self.ner_model_name == "mkgformer":
            self.ner_model = UnimoCRFModel(self.num_labels, args)
            self.vt_encoder = self.ner_model.model
        else:
            self.ner_model = None
            self.vt_encoder = None
        # Label Encoder #
        self.label_mlp = nn.Sequential(
            nn.Linear(768, self.args.label_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.label_hidden_dim, self.args.label_hidden_dim)
        )
        # Self Attentions #
        self.char_self_attn = MultiAttn(query_dim=self.args.char_hidden_dim, key_dim=self.args.char_hidden_dim, value_dim=self.args.char_hidden_dim, emb_dim=self.args.char_hidden_dim, num_heads=4, dropout_rate=0.1)
        self.label_self_attn = MultiAttn(query_dim=self.args.label_hidden_dim, key_dim=self.args.label_hidden_dim, value_dim=self.args.label_hidden_dim, emb_dim=self.args.label_hidden_dim, num_heads=1, dropout_rate=0.1)
        # Cross Attention #
        if self.ner_model_name == "hvpnet":
            self.label_vt_attn = MultiAttn(query_dim=self.args.label_hidden_dim, key_dim=self.vt_encoder.bert.config.hidden_size, value_dim=self.vt_encoder.bert.config.hidden_size, emb_dim=self.vt_encoder.bert.config.hidden_size, num_heads=4, dropout_rate=0.1)
            self.char_vt_attn = MultiAttn(query_dim=self.args.char_hidden_dim, key_dim=self.vt_encoder.bert.config.hidden_size, value_dim=self.vt_encoder.bert.config.hidden_size, emb_dim=self.vt_encoder.bert.config.hidden_size, num_heads=4, dropout_rate=0.1)
            self.vt_label_attn = MultiAttn(query_dim=self.vt_encoder.bert.config.hidden_size, key_dim=self.args.label_hidden_dim, value_dim=self.args.label_hidden_dim, emb_dim=self.args.label_hidden_dim, num_heads=4, dropout_rate=0.1)
        elif self.ner_model_name == "mkgformer":
            self.label_vt_attn = MultiAttn(query_dim=self.args.label_hidden_dim, key_dim=self.vt_encoder.text_config.hidden_size, value_dim=self.vt_encoder.text_config.hidden_size, emb_dim=self.vt_encoder.text_config.hidden_size, num_heads=4, dropout_rate=0.1)
            self.char_vt_attn = MultiAttn(query_dim=self.args.char_hidden_dim, key_dim=self.vt_encoder.text_config.hidden_size, value_dim=self.vt_encoder.text_config.hidden_size, emb_dim=self.vt_encoder.text_config.hidden_size, num_heads=4, dropout_rate=0.1)
            self.vt_label_attn = MultiAttn(query_dim=self.vt_encoder.text_config.hidden_size, key_dim=self.args.label_hidden_dim, value_dim=self.args.label_hidden_dim, emb_dim=self.args.label_hidden_dim, num_heads=4, dropout_rate=0.1)
        else:
            self.label_vt_attn = None
            self.char_vt_attn = None
            self.vt_label_attn = None
        self.char_label_attn = MultiAttn(query_dim=self.args.char_hidden_dim, key_dim=self.args.label_hidden_dim, value_dim=self.args.label_hidden_dim, emb_dim=self.args.label_hidden_dim, num_heads=4, dropout_rate=0.1)
        # Output Layers #
        if self.ner_model_name == "hvpnet":
            self.fc = nn.Sequential(
                nn.Linear(self.vt_encoder.bert.config.hidden_size + self.args.label_hidden_dim, num_labels),
                nn.LayerNorm(num_labels)
            )
        elif self.ner_model_name == "mkgformer":
            self.fc = nn.Sequential(
                nn.Linear(self.vt_encoder.text_config.hidden_size + self.args.label_hidden_dim, num_labels),
                nn.LayerNorm(num_labels)
            )
        else:
            self.fc = None
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(self.num_labels, batch_first=True)
        # Noise Scheduler #
        self.noise_scheduler = NoiseScheduler(timesteps=self.args.train_steps, device=self.args.device)

    def forward(self, t, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, rcnn_imgs=None, return_features=False):
        corrupt_label_embeddings, noise = self.corrupt(t, labels, attention_mask)
        if return_features:
            recon_targets, recon_emissions, recon_features = self.denoise(corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, return_features)
            return (recon_targets, recon_emissions, recon_features) 
        else:
            recon_targets, recon_emissions = self.denoise(corrupt_label_embeddings, t, char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, return_features)
            return recon_targets, recon_emissions
    
    def corrupt(self, t, labels=None, attention_mask=None):
        # Label Features #
        label_features = self.get_label_embedding(labels, attention_mask)
        # Add Noise #
        corrupt_label_embeddings, noise = self.noise_scheduler.add_noise(label_features, t)
        return corrupt_label_embeddings, noise
    
    def denoise(self, corrupt_label_embeddings, t, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None, return_features=False):
        t = t.float()
        time_features = torch.sin(self.time_mlp(t)).unsqueeze(1)
        char_features, vt_features = self.get_context_embedding(char_input_ids, input_ids, attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs)
        # Feature Combination #
        corrupt_label_embeddings = corrupt_label_embeddings + time_features
        char_label_features = self.char_label_attn(query=char_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attention_mask) # (bsz, len, label_hidden_dim)
        vt_label_features = self.vt_label_attn(query=vt_features, key=corrupt_label_embeddings, value=corrupt_label_embeddings, mask=attention_mask) # (bsz, len, label_hidden_dim)
        label_vt_features = self.label_vt_attn(query=corrupt_label_embeddings, key=vt_features, value=vt_features, mask=attention_mask) # (bsz, len, hidden)
        char_vt_features = self.char_vt_attn(query=char_features, key=vt_features, value=vt_features, mask=attention_mask) # (bsz, len, hidden)
        label_features_comb = char_label_features + vt_label_features
        vt_features_comb = char_vt_features + label_vt_features
        features = torch.cat((label_features_comb, vt_features_comb), dim=-1) # (bsz, len, label_hidden_dim + hidden)
        # Generate Embeddings #
        features = self.dropout(features)  # (bsz, len, label_hidden_dim + hidden)
        recon_emissions = self.fc(features) # (bsz, len, num_labels)
        recon_targets = torch.tensor(self.crf.decode(recon_emissions)).to(self.args.device)
        return (recon_targets, recon_emissions, features) if return_features else (recon_targets, recon_emissions)
    
    def get_label_embedding(self, labels=None, attention_mask=None):
        label_features = self.label_embedding_table[labels]
        label_features = self.label_mlp(label_features)
        label_features = self.label_pos_encoder(label_features)
        label_features = self.label_self_attn(query=label_features, key=label_features, value=label_features, mask=attention_mask) # (bsz, len, label_hidden_dim)
        return label_features
    
    def get_context_embedding(self, char_input_ids=None, input_ids=None, attention_mask=None, token_type_ids=None, images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = attention_mask.size(0)
        # Char Features #
        vocab_size = len(self.char_lstm.char2int)
        char_input = F.one_hot(char_input_ids, vocab_size) # (bsz, len, char_len, vocab_sz)
        char_input = char_input.view(-1, self.args.max_char_len, vocab_size).to(torch.float32) # (bsz*len, char_len, vocab_sz)
        hc = self.char_lstm.init_hidden((char_input.shape[0],))
        hc = tuple([each.to(self.args.device) for each in hc])
        _, char_hidden = self.char_lstm(char_input, hc) # char_hidden[0] (4, bsz*len, char_hidden_dim)
        char_features = char_hidden[0].transpose(0, 2).contiguous().view(bsz, self.args.max_seq_len, -1)
        char_features = self.char_pos_encoder(char_features)
        char_features = self.char_lstm_mlp(char_features)
        char_features = self.char_self_attn(query=char_features, key=char_features, value=char_features, mask=attention_mask) # (bsz, len, char_hidden_dim)
        # Visual-Textual Features #
        if self.ner_model_name == "hvpnet":
            sequence_output = self.vt_encoder(input_ids, attention_mask, token_type_ids, images, aux_imgs)
        elif self.ner_model_name == "mkgformer":
            out = self.vt_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
            sequence_output = out.last_hidden_state
        
        return char_features, sequence_output
