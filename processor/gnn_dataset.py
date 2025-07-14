import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from torch_geometric.data import HeteroData
import logging
import time
import torch.nn.functional as F
import argparse
from utils.utils import test_embedding_robustness

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(f"Number of handlers: {len(logger.handlers)}")

class NERProcessor:
    def __init__(self, args, twitter_path='data/NER_data/twitter2015', conll_path='data/NER_data/conll2003'):
        """
        Initialize processor for Twitter2015 and CoNLL-2003 NER datasets.

        Args:
            args: Argument object with local_cache_path and lm_name (e.g., 'bert-base-uncased').
            twitter_path (str): Path to Twitter2015 dataset directory.
            conll_path (str): Path to CoNLL-2003 dataset directory.
        """
        self.twitter_path = twitter_path
        self.conll_path = conll_path
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name), do_lower_case=True
        )
        self.bert = BertModel.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name)
        )
        self.LABELS = ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        self.label2word_mapping = {
            "[PAD]": "[PAD]", "O": "outside", "B-MISC": "begin miscellaneous", "I-MISC": "inside miscellaneous",
            "B-PER": "begin person", "I-PER": "inside person", "B-ORG": "begin organization", "I-ORG": "inside organization",
            "B-LOC": "begin location", "I-LOC": "inside location", "X": "unknown", "[CLS]": "[CLS]", "[SEP]": "[SEP]"
        }
        self.label_mapping = {
            "B-OTHER": "B-MISC",
            "I-OTHER": "I-MISC"
        }
        self.label_map = {label: idx for idx, label in enumerate(self.LABELS)}
        # Cache label embeddings
        start_time = time.time()
        self.label_embeddings = self.get_label_embedding()
        logger.info(f"Cached label embeddings in {time.time() - start_time:.2f} seconds")

    def load_from_file(self):
        """
        Load NER label sequences from Twitter2015 and CoNLL-2003.

        Returns:
            list: List of (tokens, labels) pairs.
        """
        start_time = time.time()
        twitter_files = [
            os.path.join(self.twitter_path, "train.txt"),
            os.path.join(self.twitter_path, "valid.txt"),
            os.path.join(self.twitter_path, "test.txt")
        ]
        conll_files = [
            os.path.join(self.conll_path, "eng.train"),
            os.path.join(self.conll_path, "eng.testa"),
            os.path.join(self.conll_path, "eng.testb")
        ]
        data = []
        sentence_count = 0
        malformed_lines = 0
        twitter_sentences = 0
        conll_sentences = 0

        def process_sequence(tokens, target, dataset_type):
            nonlocal data, sentence_count, twitter_sentences, conll_sentences
            if target and tokens:
                target = [self.label_mapping.get(t, t) for t in target]
                if all(t in self.label_map for t in target):
                    data.append((tokens, target))
                    sentence_count += 1
                    if dataset_type == "twitter":
                        twitter_sentences += 1
                    else:
                        conll_sentences += 1
                else:
                    logger.warning(f"Invalid labels in {dataset_type} sequence {sentence_count}: {target}")

        for twitter_file in twitter_files:
            if os.path.exists(twitter_file):
                logger.info(f"Loading Twitter2015 from {twitter_file}")
                tokens, target = [], []
                current_imgid = None
                with open(twitter_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("IMGID:"):
                            tokens, target = [], []
                            current_imgid = line.split("IMGID:")[1]
                        elif line:
                            if current_imgid is None:
                                logger.warning(f"Line before IMGID in {twitter_file}: {line}")
                                malformed_lines += 1
                                continue
                            parts = line.split("\t")
                            if len(parts) != 2:
                                logger.warning(f"Malformed line in IMGID:{current_imgid}: {line}")
                                malformed_lines += 1
                                continue
                            tokens.append(parts[0])
                            target.append(parts[1])
                        else:
                            process_sequence(tokens, target, "twitter")
                            tokens, target = [], []
                    process_sequence(tokens, target, "twitter")
            else:
                logger.warning(f"Twitter2015 file {twitter_file} not found")

        for conll_file in conll_files:
            if os.path.exists(conll_file):
                logger.info(f"Loading CoNLL-2003 from {conll_file}")
                tokens, target = [], []
                with open(conll_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("-DOCSTART-"):
                            continue
                        elif line:
                            parts = line.split()
                            if len(parts) < 4:
                                logger.warning(f"Malformed line in {conll_file}: {line}")
                                malformed_lines += 1
                                continue
                            tokens.append(parts[0])
                            target.append(parts[3])
                        else:
                            process_sequence(tokens, target, "conll")
                            tokens, target = [], []
                    process_sequence(tokens, target, "conll")
            else:
                logger.warning(f"CoNLL-2003 file {conll_file} not found")

        logger.info(f"Loaded {sentence_count} sequences (Twitter2015: {twitter_sentences}, CoNLL-2003: {conll_sentences}), {malformed_lines} malformed lines in {time.time() - start_time:.2f} seconds")
        return data

    def _is_valid_transition(self, src, dst):
        """
        Check if a transition is valid based on BIO rules.

        Args:
            src (str): Source label.
            dst (str): Destination label.

        Returns:
            bool: True if transition is valid.
        """
        if src == "[PAD]" or dst == "[PAD]" or src == "[CLS]" or dst == "[CLS]" or src == "[SEP]" or dst == "[SEP]":
            return False
        if src == "X" or dst == "X":
            return True
        if src == "O":
            return dst in ["O", "B-PER", "B-ORG", "B-LOC", "B-MISC"]
        if src.startswith("B-"):
            entity = src[2:]
            return dst == f"I-{entity}" or dst == "O"
        if src.startswith("I-"):
            entity = src[2:]
            return dst == f"I-{entity}" or dst == "O"
        return False

    def get_label_mapping(self):
        """Return dictionary mapping labels to indices."""
        return self.label_map

    def get_label_embedding(self):
        """
        Generate 768D BERT embeddings for labels using label2word_mapping.

        Returns:
            torch.Tensor: 768D embeddings for each label.
        """
        embeddings = []
        for label in self.LABELS:
            word = self.label2word_mapping[label]
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not token_ids or token_ids[0] == self.tokenizer.unk_token_id:
                logger.warning(f"Phrase '{word}' not in vocab, using [UNK] embedding")
                token_ids = [self.tokenizer.unk_token_id]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            logger.debug(f"Label '{label}' tokenized as: {tokens}")
            token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
            with torch.no_grad():
                embedding = self.bert.get_input_embeddings()(token_id_tensor)
                embedding = embedding.mean(dim=0)
            embeddings.append(embedding)
        return torch.stack(embeddings)

    def process(self):
        """
        Build heterogeneous graphs with nodes as BIO labels and edges based on sequence transitions, using vectorized edge creation.
        Omits data['label'].x to save memory, as label embeddings are stored in the model.

        Returns:
            list: List of HeteroData objects with 13 label nodes (no x), edge types, and sequence labels.
        """
        start_time = time.time()
        data_list = []
        sequences = self.load_from_file()
        edge_types = ['inside', 'to_entity', 'exit', 'background']
        num_sequences = len(sequences)

        for seq_idx, (tokens, labels) in enumerate(sequences):
            seq_start_time = time.time()
            if len(labels) > 128:
                logger.debug(f"Skipping sequence {seq_idx}: length {len(labels)} exceeds max_seq_len 128")
                continue

            # Validate and map labels to indices
            label_indices = [self.label_map.get(label, self.label_map['X']) for label in labels]
            if None in label_indices:
                logger.warning(f"Sequence {seq_idx}: contains invalid labels, skipping")
                continue

            # Initialize HeteroData
            data = HeteroData()
            data['label'].y = torch.tensor(label_indices, dtype=torch.long)  # Sequence labels

            # Vectorized edge creation
            if len(label_indices) > 1:
                src_indices = torch.tensor(label_indices[:-1], dtype=torch.long)
                tgt_indices = torch.tensor(label_indices[1:], dtype=torch.long)
                src_labels = labels[:-1]
                tgt_labels = labels[1:]

                # Create masks for edge types
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
                    elif src == 'O' and tgt == 'O':
                        background_mask[i] = True

                # Build edge indices for each type
                valid_edges = False
                for etype, mask in zip(edge_types, [inside_mask, to_entity_mask, exit_mask, background_mask]):
                    if mask.any():
                        edges = torch.stack([src_indices[mask], tgt_indices[mask]], dim=0)
                        # Add bidirectional edges
                        rev_edges = torch.stack([tgt_indices[mask], src_indices[mask]], dim=0)
                        edge_index = torch.cat([edges, rev_edges], dim=1)
                        if edge_index.size(1) > 0 and edge_index.max().item() < len(self.label_map):
                            data['label', etype, 'label']['edge_index'] = edge_index
                            logger.debug(f"Sequence {seq_idx} edge type {etype}: {edge_index.size(1)} edges")
                            valid_edges = True
                        else:
                            logger.warning(f"Sequence {seq_idx} edge type {etype}: invalid edge_index, max index {edge_index.max().item() if edge_index.size(1) > 0 else -1}, num_labels {len(self.label_map)}")
                    else:
                        logger.debug(f"Sequence {seq_idx} edge type {etype}: no edges")

                if valid_edges:
                    logger.info(f"Sequence {seq_idx}: y_shape={data['label'].y.shape}, y_labels={[self.LABELS[i] for i in data['label'].y]}")
                    data_list.append(data)
                else:
                    logger.debug(f"Skipping sequence {seq_idx}: no valid edges")
            else:
                logger.debug(f"Skipping sequence {seq_idx}: too short to form edges")

            if (seq_idx + 1) % 100 == 0:
                logger.info(f"Processed {seq_idx + 1}/{num_sequences} sequences, time per sequence: {(time.time() - seq_start_time):.4f}s")

        total_time = time.time() - start_time
        logger.info(f"Processed {len(data_list)} valid heterogeneous graphs in {total_time:.2f} seconds, avg {total_time / num_sequences if num_sequences > 0 else 0:.4f}s per sequence")
        return data_list

class NERDataset(Dataset):
    def __init__(self, processor, max_seq_len=128):
        """
        Initialize dataset for GNN training with label-level heterogeneous graphs.

        Args:
            processor (NERProcessor): Processor instance with loaded data.
            max_seq_len (int): Maximum sequence length for truncation.
        """
        super().__init__()
        self.processor = processor
        self.max_seq_len = max_seq_len
        start_time = time.time()
        self.data = processor.process()
        logger.info(f"NERDataset initialized in {time.time() - start_time:.2f} seconds")
        self.label_map = processor.get_label_mapping()

    def __len__(self):
        """Return number of sequences (graphs)."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return label-level heterogeneous graph data.

        Args:
            idx (int): Index of the sequence/graph.

        Returns:
            HeteroData: Graph with 13 label nodes (no x), edge types, and sequence labels, or None if invalid.
        """
        data = self.data[idx]
        if data['label'].y.size(0) > self.max_seq_len:
            logger.debug(f"Sequence {idx} exceeds max_seq_len {self.max_seq_len}")
            return None
        labels = [self.label_map.get(idx.item(), "Unknown") for idx in data['label'].y]
        logger.debug(f"Sequence {idx} labels: {labels}")
        return data

if __name__ == "__main__":
    """
    Test NERProcessor and NERDataset to verify label-level graph construction and embedding robustness.
    """
    args = argparse.Namespace(
        local_cache_path="/home/yixin/workspace/huggingface/",
        lm_name="bert-base-uncased"
    )
    semantic_similarities = {
        "B-PER": ["I-PER"], "I-PER": ["B-PER"],
        "B-ORG": ["I-ORG"], "I-ORG": ["B-ORG"],
        "B-LOC": ["I-LOC"], "I-LOC": ["B-LOC"],
        "B-MISC": ["I-MISC"], "I-MISC": ["B-MISC"],
        "O": [], "X": [], "[PAD]": [], "[CLS]": [], "[SEP]": []
    }
    logger.info("Initializing NERProcessor and NERDataset...")
    start_time = time.time()
    processor = NERProcessor(args)
    dataset = NERDataset(processor, max_seq_len=128)
    logger.info(f"Test setup completed in {time.time() - start_time:.2f} seconds")
    num_samples = min(3, len(dataset))
    inverse_label_map = {idx: label for label, idx in processor.get_label_mapping().items()}
    for idx in range(num_samples):
        sample = dataset[idx]
        if sample is None:
            logger.info(f"\nSequence {idx}: Skipped due to invalid or empty graph")
            continue
        labels = [inverse_label_map.get(idx.item(), "Unknown") for idx in sample['label'].y]
        edge_types = ['inside', 'to_entity', 'exit', 'background']
        logger.info(f"\nSequence {idx}:")
        logger.info(f"Node count: 13 (labels: {', '.join(processor.LABELS)})")
        logger.info(f"Sequence labels: {labels}")
        for etype in edge_types:
            edge_key = ('label', etype, 'label')
            if edge_key in sample.edge_types and 'edge_index' in sample[edge_key]:
                edge_list = sample[edge_key]['edge_index'].t().tolist()
                logger.info(f"Edge type {etype}: {[(inverse_label_map.get(src, 'Unknown'), inverse_label_map.get(tgt, 'Unknown')) for src, tgt in edge_list]}")
                for src_idx, dst_idx in edge_list:
                    src_label = inverse_label_map.get(src_idx, "Unknown")
                    dst_label = inverse_label_map.get(dst_idx, "Unknown")
                    assert src_label not in ["[CLS]", "[SEP]", "[PAD]"] and dst_label not in ["[CLS]", "[SEP]", "[PAD]"], \
                        f"Sequence {idx}: [CLS], [SEP], or [PAD] found in {etype} edges: {src_label} -> {dst_label}"
            else:
                logger.info(f"Edge type {etype}: No edges")
    logger.info("Label-level graph construction test passed!")
    logger.info("\nTesting BERT embedding similarities...")
    label_embeddings = processor.label_embeddings
    num_labels = label_embeddings.size(0)
    similarity_matrix = torch.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            if i != j:
                similarity = F.cosine_similarity(
                    label_embeddings[i].unsqueeze(0),
                    label_embeddings[j].unsqueeze(0),
                    dim=1
                ).item()
                similarity_matrix[i, j] = similarity
            else:
                similarity_matrix[i, j] = 1.0
    logger.info("Cosine similarity matrix:")
    for i, label_i in enumerate(processor.LABELS):
        similarities = [f"{label_j}: {similarity_matrix[i, j]:.4f}" for j, label_j in enumerate(processor.LABELS)]
        logger.info(f"{label_i}: {', '.join(similarities)}")
    max_similarity = similarity_matrix[similarity_matrix < 1.0].max().item()
    logger.info(f"Maximum cosine similarity (excluding self): {max_similarity:.4f}")
    if max_similarity >= 0.95:
        logger.warning("Some embeddings have cosine similarity >= 0.95, may not be sufficiently separated")
        for i in range(num_labels):
            for j in range(i + 1, num_labels):
                if similarity_matrix[i, j] >= 0.95:
                    logger.warning(f"High similarity between {processor.LABELS[i]} and {processor.LABELS[j]}: {similarity_matrix[i, j]:.4f}")
    else:
        logger.info("All embeddings are sufficiently separated (cosine similarity < 0.95)")
    logger.info("\nTesting embedding robustness...")
    test_embedding_robustness(
        embeddings=label_embeddings,
        labels=processor.LABELS,
        semantic_similarities=semantic_similarities,
        num_samples=1000,
        noise_levels=[0.01, 0.1, 1.0]
    )