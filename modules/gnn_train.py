import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processor.gnn_dataset import GNNProcessor, GNNDataset
from utils.utils import test_embedding_robustness
from transformers.optimization import get_linear_schedule_with_warmup
import logging
import argparse
import csv
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.data import HeteroData
from models.gnn_model import HeteroLabelEmbeddingGNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None and isinstance(item, HeteroData)]
    if not batch:
        return None
    valid_batch = []
    for idx, data in enumerate(batch):
        if 'label' not in data.node_types:
            logger.warning(f"Skipping batch item {idx}: Missing 'label' node type")
            continue
        if 'y' not in data['label'] or data['label'].y is None:
            logger.warning(f"Skipping batch item {idx}: Missing 'y' attribute for 'label' node type")
            continue
        valid_batch.append(data)
    return valid_batch if valid_batch else None

class GNNPreTrainer:
    def __init__(self, train_data, val_data, test_data, model, label_map, args):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model = model
        self.label_map = label_map
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_num_steps = len(self.train_data) * args.num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.best_robustness_score = 0.0
        self.best_dev_epoch = None
        self.no_improve = 0
        self.step = 0
        self.max_grad_norm = 1.0
        self.best_embedding_table = None
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_hetero_best_decoder.pth")
        self.semantic_similarities = {
            "B-PER": ["I-PER"], "I-PER": ["B-PER"],
            "B-ORG": ["I-ORG"], "I-ORG": ["B-ORG"],
            "B-LOC": ["I-LOC"], "I-LOC": ["B-LOC"],
            "B-MISC": ["I-MISC"], "I-MISC": ["B-MISC"],
            "O": [], "X": [], "[PAD]": [], "[CLS]": [], "[SEP]": []
        }
        os.makedirs(args.save_path, exist_ok=True)
        if args.metrics_file:
            with open(self.args.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'stage', 'batch', 'loss', 'decoder_loss', 'locality_1.0', 'clusteredness_1.0', 'separation'])

    def decoder_loss(self, embeddings, logits, label_indices, clean_weight=0.5, sigma=0.5, num_noise_samples=5):
        if logits is None or label_indices.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        clean_loss = F.cross_entropy(logits, label_indices, reduction='mean')
        # Select embeddings for the sequence's tokens
        seq_embeddings = embeddings[label_indices]  # Shape: [seq_len, hidden_dim]
        noisy_embeddings = seq_embeddings.unsqueeze(1) + sigma * torch.randn(
            seq_embeddings.size(0), num_noise_samples, seq_embeddings.size(1), device=self.device
        )  # Shape: [seq_len, num_noise_samples, hidden_dim]
        noisy_embeddings = noisy_embeddings.view(-1, seq_embeddings.size(1))  # Shape: [seq_len * num_noise_samples, hidden_dim]
        noisy_labels = label_indices.repeat_interleave(num_noise_samples)  # Shape: [seq_len * num_noise_samples]
        noisy_logits = self.model.decoder(noisy_embeddings)  # Shape: [seq_len * num_noise_samples, num_labels]
        noisy_loss = F.cross_entropy(noisy_logits, noisy_labels, reduction='mean')
        total_loss = clean_weight * clean_loss + (1.0 - clean_weight) * noisy_loss
        return total_loss

    def _step(self, batch, stage="train"):
        if batch is None or not batch or all(not isinstance(data, HeteroData) for data in batch):
            logger.warning("Invalid batch, skipping")
            return None
        embeddings = []
        logits = []
        label_indices_list = []
        for data in batch:
            if not isinstance(data, HeteroData):
                continue
            edge_index_dict = {
                k: data[k]['edge_index'].to(self.device)
                for k in data.edge_types if 'edge_index' in data[k] and isinstance(data[k]['edge_index'], torch.Tensor)
            }
            label_indices = data['label'].y.to(self.device)
            if stage == "train":
                emb, _, log = self.model(edge_index_dict, label_indices)
            else:
                with torch.no_grad():
                    emb, _, log = self.model(edge_index_dict, label_indices)
            if emb is None or log is None:
                continue
            embeddings.append(emb)
            logits.append(log)
            label_indices_list.append(label_indices)
        if not embeddings:
            logger.warning("No valid embeddings, skipping")
            return None
        embeddings = torch.stack(embeddings, dim=0)
        mean_embeddings = embeddings.mean(dim=0)
        valid_logits = torch.cat(logits, dim=0)
        valid_label_indices = torch.cat(label_indices_list, dim=0)
        loss = self.decoder_loss(mean_embeddings, valid_logits, valid_label_indices, clean_weight=self.args.clean_weight)
        return (loss, [mean_embeddings], label_indices_list)

    def train(self, epochs=None):
        epochs = epochs or self.args.num_epochs
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0.0
                total_samples = 0
                pbar.set_description_str(f"Epoch {epoch + 1}/{epochs}")
                epoch_start_time = time.time()
                for batch in self.train_data:
                    self.step += 1
                    if not batch or all(not isinstance(data, HeteroData) for data in batch):
                        continue
                    result = self._step(batch, stage="train")
                    if result is None:
                        continue
                    loss, _, label_indices_list = result
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.step % self.args.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                    batch_samples = sum(len(n) for n in label_indices_list)
                    total_samples += batch_samples
                    total_loss += loss.item() * batch_samples
                    if self.step % self.args.refresh_step == 0:
                        pbar.update(self.args.refresh_step)
                        pbar.set_postfix_str(f"loss: {total_loss / total_samples if total_samples > 0 else 0.0:<6.5f}")
                if total_samples > 0:
                    avg_loss = total_loss / total_samples
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
                               f"Decoder Loss: {avg_loss:.4f}, "
                               f"Time: {time.time() - epoch_start_time:.2f}s")
                    if self.args.metrics_file:
                        with open(self.args.metrics_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1, "train", len(self.train_data),
                                avg_loss, avg_loss,
                                0.0, 0.0, 0.0
                            ])
                if epoch >= self.args.eval_begin_epoch:
                    _, _, _, _, early_stop = self.evaluate(epoch, stage="val")
                    if early_stop:
                        break
            torch.cuda.empty_cache()
            pbar.close()
        return self.best_embedding_table if self.best_embedding_table is not None else None

    def evaluate(self, epoch, stage="val"):
        self.model.eval()
        num_labels = len(self.label_map)
        hidden_dim = self.model.decoder.in_features
        embedding_sums = torch.zeros(num_labels, hidden_dim).to(self.device)
        embedding_counts = torch.zeros(num_labels).to(self.device)
        total_loss = 0.0
        total_samples = 0
        data_loader = self.val_data if stage == "val" else self.test_data
        for batch in data_loader:
            if not batch:
                continue
            result = self._step(batch, stage=stage)
            if result is None:
                continue
            loss, embeddings, label_indices_list = result
            mean_embeddings = embeddings[0]
            batch_samples = sum(len(n) for n in label_indices_list)
            total_samples += batch_samples
            total_loss += loss.item() * batch_samples
            for label_idx in range(num_labels):
                embedding_sums[label_idx] += mean_embeddings[label_idx]
                embedding_counts[label_idx] += 1
        embedding_table = embedding_sums / (embedding_counts.unsqueeze(1) + 1e-8)
        results = test_embedding_robustness(
            embeddings=embedding_table,
            labels=list(self.label_map.keys()),
            semantic_similarities=self.semantic_similarities,
            num_samples=1000,
            noise_levels=[0.01, 0.1, 1.0]
        )
        locality_1_0 = results[1.0]["locality"]
        clusteredness_1_0 = results[1.0]["clusteredness"]
        separation = results[1.0]["separation"]
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info(f"{stage.capitalize()} Loss: {avg_loss:.4f}, "
                   f"Decoder Loss: {avg_loss:.4f}")
        logger.info(f"{stage.capitalize()} Robustness Metrics (sigma=1.0): Locality: {locality_1_0:.4f}, "
                   f"Clusteredness: {clusteredness_1_0:.4f}, Separation: {separation:.4f}")
        if self.args.metrics_file:
            with open(self.args.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, stage, len(data_loader),
                    avg_loss, avg_loss,
                    locality_1_0, clusteredness_1_0, separation
                ])
        early_stop = False
        if stage == "val":
            robustness_score = (locality_1_0 + clusteredness_1_0) / 2
            if robustness_score > self.best_robustness_score:
                self.best_robustness_score = robustness_score
                self.best_dev_epoch = epoch + 1
                self.no_improve = 0
                self.best_embedding_table = embedding_table
                if self.args.save_path:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    logger.info(f"Saved best model (Robustness Score: {robustness_score:.4f}) to {self.best_model_path}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.args.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    early_stop = True
        return embedding_table, locality_1_0, clusteredness_1_0, separation, early_stop

    def test(self, epoch=0):
        if os.path.exists(self.best_model_path):
            logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            logger.info("Load model successful!")
        else:
            logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")
        embedding_table, _, _, _, _ = self.evaluate(epoch, stage="test")
        return embedding_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Heterogeneous GNN for NER embeddings")
    parser.add_argument("--local_cache_path", type=str, default="/home/yixin/workspace/huggingface/", help="Path to cache")
    parser.add_argument("--lm_name", type=str, default="bert-base-uncased", help="Language model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio for scheduler")
    parser.add_argument("--eval_begin_epoch", type=int, default=1, help="Epoch to start validation")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Path to save models")
    parser.add_argument("--dataset_name", type=str, default="gnn", help="Dataset name for checkpoint files")
    parser.add_argument("--metrics_file", type=str, default="metrics.csv", help="File to save metrics")
    parser.add_argument("--refresh_step", type=int, default=2, help="Steps to update progress bar")
    parser.add_argument("--clean_weight", type=float, default=0.5, help="Weight for clean loss in decoder_loss")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The dimension of embeddngs.")
    parser.add_argument('--max_seq_len', default=80, type=int, help="Max sequence length.")
    args = parser.parse_args()
    logger.info("Initializing GNNProcessor and GNNDataset...")
    processor = GNNProcessor(args)
    dataset = GNNDataset(processor)
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)
    logger.info("Initializing Heterogeneous GNN model...")
    model = HeteroLabelEmbeddingGNN(label_embeddings=processor.label_embeddings, hidden_dim=args.hidden_dim, num_labels=len(processor.get_label_mapping()))
    logger.info("Training Heterogeneous GNN with decoder loss...")
    trainer = GNNPreTrainer(
        train_data=train_loader,
        val_data=val_loader,
        test_data=test_loader,
        model=model,
        label_map=processor.get_label_mapping(),
        args=args
    )
    embedding_table = trainer.train()
    logger.info("Testing on test set...")
    test_embedding_table = trainer.test()