import torch
import numpy as np
import random
from torch import nn
import albumentations as A
from PIL import Image
import logging
from tqdm import tqdm
from collections import Counter

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_seed(seed=2021):
    """sets random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Weak augmentation #
def remove_excluded_words(words, labels, excluded_words):
    """
    Remove tokens that are in the excluded_tokens list and their corresponding labels.

    Args:
        tokens (List[str]): List of tokens in a sentence.
        labels (List[str]): Corresponding list of labels for the tokens.
        excluded_tokens (List[str]): List of tokens that should be removed.

    Returns:
        Tuple[List[str], List[str]]: Filtered tokens and their labels.
    """
    assert len(words) == len(labels), "Tokens and labels must have the same length."

    # Filter out excluded tokens and their labels
    filtered_words = [word for word in words if word not in excluded_words]
    filtered_labels = [label for word, label in zip(words, labels) if word not in excluded_words]

    return filtered_words, filtered_labels

# Strong augmentation #
def strong_augment_pil_image(image):
    """
    Apply strong augmentations to a PIL image using Albumentations.

    Args:
        pil_image (Image.Image): Input PIL image.

    Returns:
        Image.Image: Augmented PIL image.
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Define Albumentations transformations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    ])

    # Apply the transformations
    augmented = transform(image=image_np)
    augmented_image_np = augmented["image"]

    # Convert NumPy array back to PIL image
    augmented_image = Image.fromarray(augmented_image_np)

    return augmented_image

def test_embedding_robustness(embeddings, labels, semantic_similarities, num_samples=1000, noise_levels=[0.01, 0.1, 1.0]):
    """
    Test whether embeddings satisfy locality, clusteredness, and separation under Gaussian noise.

    Args:
        embeddings (torch.Tensor): Embedding table, shape [num_labels, embedding_dim].
        labels (list): List of label names corresponding to embeddings.
        semantic_similarities (dict): Dictionary mapping each label to a list of semantically close labels.
        num_samples (int): Number of noisy samples per label to generate.
        noise_levels (list): List of standard deviations for Gaussian noise.

    Returns:
        dict: Metrics for locality, clusteredness, and separation per noise level.
    """
    device = embeddings.device
    num_labels = len(labels)
    label_map = {label: idx for idx, label in enumerate(labels)}
    inverse_label_map = {idx: label for label, idx in label_map.items()}
    results = {sigma: {"locality": [], "clusteredness": [], "separation": []} for sigma in noise_levels}

    # Exclude special tokens from separation calculation
    valid_indices = [idx for idx, label in enumerate(labels) if label not in ["[PAD]", "[CLS]", "[SEP]"]]
    valid_embeddings = embeddings[valid_indices]
    valid_labels = [labels[idx] for idx in valid_indices]

    logger.info("Testing embedding robustness under Gaussian noise...")
    for sigma in noise_levels:
        logger.info(f"Testing noise level sigma={sigma}")
        correct_counts = torch.zeros(num_labels, dtype=torch.float, device=device)
        confusion_matrix = torch.zeros(num_labels, num_labels, dtype=torch.float, device=device)
        locality_correct = 0
        total_samples = 0

        for label_idx, label in enumerate(labels):
            if label in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            embedding = embeddings[label_idx]
            noise = torch.randn(num_samples, embedding.size(0), device=device) * sigma
            noisy_embeddings = embedding + noise

            distances = torch.cdist(noisy_embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)
            nearest_indices = distances.argmin(dim=1)

            correct = (nearest_indices == label_idx).float().sum()
            correct_counts[label_idx] = correct / num_samples

            for pred_idx in nearest_indices:
                confusion_matrix[label_idx, pred_idx] += 1 / num_samples

            for pred_idx in nearest_indices:
                pred_label = inverse_label_map[pred_idx.item()]
                if pred_idx == label_idx or pred_label in semantic_similarities[label]:
                    locality_correct += 1
                total_samples += 1

        clusteredness = correct_counts[valid_indices].mean().item() if valid_indices else 0.0
        locality = locality_correct / total_samples if total_samples > 0 else 0.0
        pairwise_distances = torch.cdist(valid_embeddings, valid_embeddings)
        pairwise_distances.fill_diagonal_(float('inf'))
        min_distance = pairwise_distances.min().item()
        min_indices = torch.where(pairwise_distances == min_distance)
        min_pairs = [(valid_labels[i.item()], valid_labels[j.item()]) for i, j in zip(min_indices[0], min_indices[1])]
        logger.debug(f"Min distance pairs: {min_pairs}, distance: {min_distance:.4f}")

        results[sigma]["locality"] = locality
        results[sigma]["clusteredness"] = clusteredness
        results[sigma]["separation"] = min_distance

        logger.info(f"Sigma={sigma}:")
        logger.info(f"  Locality: {locality:.4f} (fraction decoding to correct or semantically close label)")
        logger.info(f"  Clusteredness: {clusteredness:.4f} (avg accuracy of denoising to correct label)")
        logger.info(f"  Separation: {min_distance:.4f} (min Euclidean distance between distinct labels)")
        logger.info(f"  Confusion matrix:")
        for i, label in enumerate(labels):
            counts = [f"{labels[j]}: {confusion_matrix[i, j]:.4f}" for j in range(num_labels)]
            logger.info(f"    {label}: {', '.join(counts)}")

        if locality < 0.9:
            logger.warning(f"Locality failure at sigma={sigma}: {locality:.4f} < 0.9")
        if clusteredness < 0.9:
            logger.warning(f"Clusteredness failure at sigma={sigma}: {clusteredness:.4f} < 0.9")
        if min_distance < 1.0:
            logger.warning(f"Separation failure: Min distance {min_distance:.4f} < 1.0")

    return results
        
# class LabelEmbeddingNormalizer(nn.Module):
#     def __init__(self, label_encoder, num_labels=13, eps=1e-6, device="cpu"):
#         """
#         Args:
#             label_encoder: function or module that maps label_indices -> embeddings
#             num_labels: total number of labels in the label table
#             eps: small constant for numerical stability
#             device: device to store buffers
#         """
#         super().__init__()
#         self.label_encoder = label_encoder
#         self.num_labels = num_labels
#         self.eps = eps
#         self.device = device

#         # will be filled by adapt()
#         self.register_buffer("mean", None)
#         self.register_buffer("std", None)

#     @torch.no_grad()
#     def adapt(self, dataloader=None, mode="dataset"):
#         """
#         Compute mean/std from label embeddings.

#         Args:
#             dataloader: required if mode="dataset" (for label frequencies)
#             mode: "table" (all labels equal) or "dataset" (weighted by label frequencies)
#         """
#         # --- Step 1: precompute embeddings for all labels ---
#         label_indices = torch.arange(self.num_labels, device=self.device).unsqueeze(0)
#         _, label_embs, _ = self.label_encoder(edge_index_dict=None, label_indices=label_indices)
#         label_embs = label_embs.squeeze(0).to(self.device)
#         # shape: (num_labels, dim)

#         if mode == "table":
#             # Equal weight across labels
#             mean = label_embs.mean(dim=0)
#             std = label_embs.std(dim=0, unbiased=False)

#         elif mode == "dataset":
#             if dataloader is None:
#                 raise ValueError("dataloader must be provided for mode='dataset'")

#             # Count frequencies
#             label_counter = Counter()
#             with tqdm(total=len(dataloader), leave=False, dynamic_ncols=True, desc="Counting Labels") as pbar:
#                 for batch in dataloader:
#                     labels, *_ = batch  # assume labels shape: (bsz, seq_len)
#                     labels = labels.view(-1).tolist()
#                     label_counter.update(labels)
#                     pbar.update()

#                 freqs = torch.tensor([label_counter[i] for i in range(self.num_labels)],
#                                     dtype=torch.float32, device=self.device)
#                 freqs = freqs / freqs.sum()  # normalize to probabilities
#                 pbar.close()

#             # Weighted stats
#             mean = (freqs.unsqueeze(1) * label_embs).sum(dim=0)
#             diffs = label_embs - mean
#             var = (freqs.unsqueeze(1) * (diffs ** 2)).sum(dim=0)
#             std = torch.sqrt(var + self.eps)
#             print(mean)
#             print(std)

#         else:
#             raise ValueError("mode must be 'table' or 'dataset'")

#         # Save buffers
#         self.mean = mean
#         self.std = std

#     def forward(self, label_embs):
#         """Normalize embeddings (bsz, seq_len, dim)."""
#         if self.mean is None or self.std is None:
#             raise RuntimeError("You must call .adapt(...) before using forward()")
#         return (label_embs - self.mean) / (self.std + self.eps)

#     def denormalize(self, norm_embs):
#         """Inverse of forward()."""
#         if self.mean is None or self.std is None:
#             raise RuntimeError("You must call .adapt(...) before using denormalize()")
#         return norm_embs * (self.std + self.eps) + self.mean

# class LabelEmbeddingNormalizer(nn.Module):
#     def __init__(self, label_encoder=None, eps=1e-6, device="cpu"):
#         super().__init__()
#         self.label_encoder = label_encoder
#         self.eps = eps
#         self.device = device

#         # Final stats
#         self.register_buffer("mean", None)
#         self.register_buffer("std", None)

#     def adapt(self, data_loader):
#         """
#         Incrementally update mean and std using data_loader.
#         Can be called multiple times on train/val/test loaders.
#         """
#         sum_ = torch.zeros(128, device=self.device)
#         sumsq = torch.zeros(128, device=self.device)
#         n_embs = 0

#         with tqdm(total=len(data_loader), leave=False, dynamic_ncols=True, desc="Computing") as pbar:
#             for batch in data_loader:
#                 batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
#                 labels, *_ = batch
#                 _, label_embs, _ = self.label_encoder(edge_index_dict=None, label_indices=labels)
#                 label_embs = label_embs.reshape(-1, label_embs.shape[-1]).to(self.device)  # (N, dim)
#                 # Update accumulators
#                 sum_ += label_embs.sum(dim=0)
#                 sumsq += (label_embs ** 2).sum(dim=0)
#                 n_embs += label_embs.shape[0]
#                 pbar.update()

#             # Update stats after this call
#             mean = sum_ / n_embs
#             var = (sumsq / n_embs) - mean.pow(2)
#             std = torch.sqrt(var + self.eps)
#             pbar.close()
#         print(mean)
#         print(std)

#         self.mean = mean
#         self.std = std

#     def forward(self, label_embs):
#         if self.mean is None or self.std is None:
#             raise RuntimeError("You must call .adapt(dataset) before using forward().")
#         return (label_embs - self.mean) / (self.std + self.eps)

#     def denormalize(self, label_embs):
#         if self.mean is None or self.std is None:
#             raise RuntimeError("You must call .adapt(dataset) before using denormalize().")
#         return label_embs * (self.std + self.eps) + self.mean

class LabelEmbeddingNormalizer(nn.Module):
    def __init__(self, num_labels, label_encoder=None, eps=1e-6, device="cpu"):
        super().__init__()
        self.num_labels = num_labels
        self.label_encoder = label_encoder
        self.eps = eps
        self.device = device

        # Buffers for per-label mean/std
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

        # Also keep global mean/std for fallback use
        self.register_buffer("global_mean", None)
        self.register_buffer("global_std", None)

    def adapt(self, data_loader):
        """
        Compute per-label mean/std and global mean/std from embeddings.
        """
        sum_, sumsq, counts = None, None, None

        with tqdm(total=len(data_loader), leave=False, dynamic_ncols=True, desc="Computing") as pbar:
            for batch in data_loader:
                batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
                labels, *_ = batch  # (B, L)

                # Forward pass through label encoder
                _, label_embs, _ = self.label_encoder(edge_index_dict=None, label_indices=labels)
                B, L, D = label_embs.shape
                label_embs = label_embs.reshape(-1, D).to(self.device)  # (N, D)
                labels = labels.reshape(-1)  # (N,)

                if sum_ is None:
                    sum_ = torch.zeros(self.num_labels, D, device=self.device)
                    sumsq = torch.zeros(self.num_labels, D, device=self.device)
                    counts = torch.zeros(self.num_labels, device=self.device)

                # Accumulate stats per label
                for lbl in labels.unique():
                    mask = (labels == lbl)
                    if mask.any():
                        embs_lbl = label_embs[mask]
                        sum_[lbl] += embs_lbl.sum(dim=0)
                        sumsq[lbl] += (embs_lbl ** 2).sum(dim=0)
                        counts[lbl] += mask.sum().item()
                pbar.update()
        pbar.close()

        # Finalize per-label stats
        mean = torch.zeros_like(sum_)
        std = torch.ones_like(sum_)
        for lbl in range(self.num_labels):
            if counts[lbl] > 0:
                mean[lbl] = sum_[lbl] / counts[lbl]
                var = (sumsq[lbl] / counts[lbl]) - mean[lbl].pow(2)
                std[lbl] = torch.sqrt(var + self.eps)

        # Global stats (just average across labels, weighted by counts)
        total_count = counts.sum()
        global_mean = (sum_.sum(dim=0) / total_count)
        global_var = (sumsq.sum(dim=0) / total_count) - global_mean.pow(2)
        global_std = torch.sqrt(global_var + self.eps)

        # Store as detached buffers
        self.register_buffer("mean", mean.detach())
        self.register_buffer("std", std.detach())
        self.register_buffer("global_mean", global_mean.detach())
        self.register_buffer("global_std", global_std.detach())

        print("Per-label mean/std shape:", mean.shape)
        print("Global mean/std shape:", global_mean.shape)

    def forward(self, label_embs, labels=None, probs=None):
        """
        Normalize embeddings.
        Args:
            label_embs: (B, L, D)
            labels: (B, L) or None (if labels not available)
            probs: (B, L, num_labels) or None (soft label predictions)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("You must call .adapt(dataset) before using forward().")

        if labels is not None:
            # Supervised: use per-label stats
            mean = self.mean[labels]  # (B, L, D)
            std = self.std[labels]    # (B, L, D)
        elif probs is not None:
            # Soft label predictions: weighted combination
            mean = torch.einsum("blc,cd->bld", probs, self.mean)  # (B, L, D)
            std = torch.einsum("blc,cd->bld", probs, self.std)    # (B, L, D)
        else:
            # Fallback: global stats
            mean = self.global_mean.view(1, 1, -1)
            std = self.global_std.view(1, 1, -1)

        return (label_embs - mean) / (std + self.eps)

    def denormalize(self, label_embs, labels=None, probs=None):
        """
        Revert normalization.
        Args:
            label_embs: (B, L, D)
            labels: (B, L) or None
            probs: (B, L, num_labels) or None
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("You must call .adapt(dataset) before using denormalize().")

        if labels is not None:
            mean = self.mean[labels]
            std = self.std[labels]
        elif probs is not None:
            mean = torch.einsum("blc,cd->bld", probs, self.mean)
            std = torch.einsum("blc,cd->bld", probs, self.std)
        else:
            mean = self.global_mean.view(1, 1, -1)
            std = self.global_std.view(1, 1, -1)

        return label_embs * (std + self.eps) + mean