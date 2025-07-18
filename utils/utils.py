import torch
import numpy as np
import random
from torch import nn
from collections import OrderedDict
from torch.utils.data import Subset
import albumentations as A
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import logging

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