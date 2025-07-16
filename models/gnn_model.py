import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class HeteroLabelEmbeddingGNN(nn.Module):
    def __init__(self, label_embeddings, hidden_dim=32, num_labels=13):
        super().__init__()
        self.register_buffer('label_embeddings', label_embeddings)
        self.projection = nn.Linear(label_embeddings.size(1), hidden_dim)
        self.convs = nn.ModuleList([
            HeteroConv({
                ('label', 'inside', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'background', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'to_entity', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'exit', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            }, aggr='mean'),
            HeteroConv({
                ('label', 'inside', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'background', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'to_entity', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ('label', 'exit', 'label'): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            }, aggr='mean')
        ])
        self.decoder = nn.Linear(hidden_dim, num_labels)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.label_map = {
            "[PAD]": 0, "O": 1, "B-MISC": 2, "I-MISC": 3, "B-PER": 4, "I-PER": 5,
            "B-ORG": 6, "I-ORG": 7, "B-LOC": 8, "I-LOC": 9, "X": 10, "[CLS]": 11, "[SEP]": 12
        }
        self.inverse_label_map = {idx: label for label, idx in self.label_map.items()}

    def _create_edge_index_dict(self, label_indices):
        """
        Create edge_index_dict from batched label indices for diffusion model training.
        Args:
            label_indices (torch.Tensor): Shape [batch_size, seq_len]
        Returns:
            list: List of edge_index_dict for each sequence in the batch
        """
        batch_size, seq_len = label_indices.size()
        edge_index_dicts = []
        for b in range(batch_size):
            indices = label_indices[b]
            if seq_len < 2:
                logger.debug(f"Batch {b}: Sequence too short: {seq_len} tokens")
                edge_index_dicts.append(None)
                continue
            labels = [self.inverse_label_map.get(idx.item(), 'X') for idx in indices]
            edge_types = ['inside', 'to_entity', 'exit', 'background']
            edge_index_dict = {}
            src_indices = indices[:-1]
            tgt_indices = indices[1:]
            src_labels = labels[:-1]
            tgt_labels = labels[1:]

            inside_mask = torch.zeros(len(src_indices), dtype=torch.bool)
            to_entity_mask = torch.zeros(len(src_indices), dtype=torch.bool)
            exit_mask = torch.zeros(len(src_indices), dtype=torch.bool)
            background_mask = torch.zeros(len(src_indices), dtype=torch.bool)

            for i, (src, tgt) in enumerate(zip(src_labels, tgt_labels)):
                if src.startswith('B-') and tgt.startswith('I-') and src[2:] == tgt[2:]:
                    inside_mask[i] = True
                elif src.startswith('I-') and tgt.startswith('I-') and src[2:] == tgt[2:]:
                    inside_mask[i] = True
                elif src == 'O' and tgt.startswith('B-'):
                    to_entity_mask[i] = True
                elif src.startswith(('B-', 'I-')) and tgt == 'O':
                    exit_mask[i] = True
                else:
                    background_mask[i] = True

            for etype, mask in zip(edge_types, [inside_mask, to_entity_mask, exit_mask, background_mask]):
                if mask.any():
                    edges = torch.stack([src_indices[mask], tgt_indices[mask]], dim=0)
                    rev_edges = torch.stack([tgt_indices[mask], src_indices[mask]], dim=0)
                    edge_index = torch.cat([edges, rev_edges], dim=1)
                    if edge_index.size(1) > 0 and edge_index.max().item() < len(self.label_map):
                        edge_index_dict[('label', etype, 'label')] = edge_index
            edge_index_dicts.append(edge_index_dict if edge_index_dict else None)
        return edge_index_dicts

    def forward(self, edge_index_dict=None, label_indices=None):
        """
        Forward pass for the GNN.
        Args:
            edge_index_dict (dict, optional): Dictionary of edge indices (pre-training).
            label_indices (torch.Tensor, optional): Shape [batch_size, seq_len] or [seq_len] (joint training or pre-training).
        Returns:
            tuple: (embeddings, sequence_embeddings, logits)
                   - Pre-training: (embeddings, None, logits)
                   - Joint training: (None, sequence_embeddings, None)
        """
        x_dict = {'label': self.relu(self.projection(self.label_embeddings))}
        is_pretraining = edge_index_dict is not None

        if is_pretraining:
            edge_index_dict = {k: v.to(self.label_embeddings.device) for k, v in edge_index_dict.items()}
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {'label': self.relu(self.dropout(x_dict['label']))}
            x_dict = {'label': self.norm(x_dict['label'])}
            embeddings = x_dict['label']
            if label_indices is not None:
                if label_indices.dim() == 2:
                    label_indices = label_indices.squeeze(0)
                logits = self.decoder(embeddings[label_indices])
                return embeddings, None, logits
            return embeddings, None, None
        else:
            if label_indices is None:
                return None, None, None
            edge_index_dicts = self._create_edge_index_dict(label_indices)
            sequence_embeddings = []
            for i, edge_index_dict in enumerate(edge_index_dicts):
                if edge_index_dict is None:
                    seq_len = label_indices.size(1)
                    sequence_embeddings.append(torch.zeros(seq_len, self.projection.out_features, device=self.label_embeddings.device))
                    continue
                edge_index_dict = {k: v.to(self.label_embeddings.device) for k, v in edge_index_dict.items()}
                x_dict_b = {'label': x_dict['label'].clone()}
                for conv in self.convs:
                    x_dict_b = conv(x_dict_b, edge_index_dict)
                    x_dict_b = {'label': self.relu(self.dropout(x_dict_b['label']))}
                x_dict_b = {'label': self.norm(x_dict_b['label'])}
                seq_emb = x_dict_b['label'][label_indices[i]]
                sequence_embeddings.append(seq_emb)
            sequence_embeddings = torch.stack(sequence_embeddings)
            return None, sequence_embeddings, None