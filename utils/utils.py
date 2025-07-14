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

# def seq_to_mask(seq_len, max_len):
#     """[get attention mask with sequence length]

#     Args:
#         seq_len ([torch.tensor]): [shape: bsz, each sequence length in a batch]
#     """
#     max_len = int(max_len) if max_len else seq_len.max().long()
#     cast_seq = torch.arange(max_len).expand(seq_len.size(0), -1).to(seq_len)
#     mask = cast_seq.lt(seq_len.unsqueeze(1))
#     return mask


def set_seed(seed=2021):
    """sets random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# def convert_preds_to_outputs(preds, raw_words, mapping, tokenizer):
#     """convet model predicitons to BIO outputs

#     Args:
#         preds ([torch.Tensor]): [prompt model predictions, (bsz x seq_len x labels)]
#         raw_words ([List]): [source raw words]
#         mapping ([dict]): [map entity labels to <<>>]
#         tokenizer : [BartTokenizer]

#     Returns:
#         [outputs (List)]: [each item length equal to raw_words, BIO format.]
#     """
#     id2label = list(mapping.keys())
#     pred_eos_index = preds.flip(dims=[1]).eq(1).cumsum(dim=1).long()
#     preds = preds[:, 1:]
#     pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
#     pred_seq_len = (pred_seq_len - 2).tolist()

#     word_start_index = len(mapping) + 2
#     outputs = []
#     for i, pred_item in enumerate(preds.tolist()):
#         pred_item = pred_item[:pred_seq_len[i]] # single sentence prediction
#         pairs, cur_pair = [], []
#         if len(pred_item):  # this sentence prediciton= is not null
#             for idx in pred_item:
#                 if idx < word_start_index:  # is entity
#                     if len(cur_pair) > 0:
#                         # assert word[i] < word[i+1]
#                         if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
#                             pairs.append(tuple(cur_pair + [idx]))   # add valid words and current entity id
#                     cur_pair = []   # clear word pairs
#                 else:   # is word
#                     cur_pair.append(idx)    # add word id to word pairs
#         raw_words_item = raw_words[i]
#         cum_lens = [1]
#         start_idx = 1
#         for word in raw_words_item:
#             start_idx += len(tokenizer.tokenize(word, add_prefix_space=True))
#             cum_lens.append(start_idx)
#         cum_lens.append(start_idx+1)
#         output = ['O' for _ in range(len(raw_words_item))]
#         # pairs: List[(word id, ... , entity id), (...), ...]
#         for pair in pairs:  # (word id, ... , entity id)
#             entity = pair[-1]
#             words = []
#             for word in pair[:-1]:
#                 if word-word_start_index in cum_lens:
#                     words.append(cum_lens.index(word-word_start_index)) 
#             if len(words) == 0: continue
#             start_idx = words[0]
#             end_idx = words[-1]
#             output[start_idx] = f'B-{id2label[entity-2]}'
#             for _ in range(start_idx+1, end_idx+1):
#                 output[_] = f'I-{id2label[entity-2]}'
#         outputs.append(output)
#     return outputs


# def write_predictions(path, texts, labels, imgids=None):
#     """[write model predictions to path (conll format)]

#     Args:
#         path ([str]): [save path]
#         texts ([List]): [raw texts]
#         labels ([List]): [predict labels]
#     """
#     print(len(texts), len(labels))
#     assert len(texts) == len(labels)
#     with open(path, "w", encoding="utf-8") as f:
#         # f.writelines("-DOCSTART-	O\n\n")
#         for i in range(len(texts)):
#             if imgids is not None:
#                 f.writelines("IMGID:{}\n".format(imgids[i]))
#             for j in range(len(texts[i])):
#                 f.writelines("{}\t{}\n".format(texts[i][j], labels[i][j].upper()))
#             f.writelines("\n")


# def write_bert_predictions(path, labels):
#     """[write model predictions to path (conll format)]

#     Args:
#         path ([str]): [save path]
#         labels ([List]): [predict labels]
#     """
#     with open(path, "w", encoding="utf-8") as f:
#         for i in range(len(labels)):
#             for j in range(len(labels[i])):
#                 f.writelines(labels[i][j].upper())
#             f.writelines("\n")


# def summary(model, *inputs, batch_size=-1, show_input=True):
#     '''
#     打印模型结构信息
#     :param model:
#     :param inputs:
#     :param batch_size:
#     :param show_input:
#     :return:
#     Example:
#         >>> print("model summary info: ")
#         >>> for step,batch in enumerate(train_data):
#         >>>     summary(self.model,*batch,show_input=True)
#         >>>     break
#     '''

#     def register_hook(module):
#         def hook(module, input, output=None):
#             class_name = str(module.__class__).split(".")[-1].split("'")[0]
#             module_idx = len(summary)
#             m_key = f"{class_name}-{module_idx + 1}"
#             summary[m_key] = OrderedDict()
#             summary[m_key]["input_shape"] = list(input[0].size())
#             summary[m_key]["input_shape"][0] = batch_size

#             if show_input is False and output is not None:
#                 if isinstance(output, (list, tuple)):
#                     for out in output:
#                         if isinstance(out, torch.Tensor):
#                             summary[m_key]["output_shape"] = [
#                                 [-1] + list(out.size())[1:]
#                             ][0]
#                         else:
#                             summary[m_key]["output_shape"] = [
#                                 [-1] + list(out[0].size())[1:]
#                             ][0]
#                 else:
#                     summary[m_key]["output_shape"] = list(output.size())
#                     summary[m_key]["output_shape"][0] = batch_size

#             params = 0
#             if hasattr(module, "weight") and hasattr(module.weight, "size"):
#                 params += torch.prod(torch.LongTensor(list(module.weight.size())))
#                 summary[m_key]["trainable"] = module.weight.requires_grad
#             if hasattr(module, "bias") and hasattr(module.bias, "size"):
#                 params += torch.prod(torch.LongTensor(list(module.bias.size())))
#             summary[m_key]["nb_params"] = params

#         if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
#             if show_input is True:
#                 hooks.append(module.register_forward_pre_hook(hook))
#             else:
#                 hooks.append(module.register_forward_hook(hook))

#     # create properties
#     summary = OrderedDict()
#     hooks = []

#     # register hook
#     model.apply(register_hook)
#     model(*inputs)

#     # remove these hooks
#     for h in hooks:
#         h.remove()

#     print("-----------------------------------------------------------------------")
#     if show_input is True:
#         line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
#     else:
#         line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
#     print(line_new)
#     print("=======================================================================")

#     total_params = 0
#     total_output = 0
#     trainable_params = 0
#     for layer in summary:
#         # input_shape, output_shape, trainable, nb_params
#         if show_input is True:
#             line_new = "{:>25}  {:>25} {:>15}".format(
#                 layer,
#                 str(summary[layer]["input_shape"]),
#                 "{0:,}".format(summary[layer]["nb_params"]),
#             )
#         else:
#             line_new = "{:>25}  {:>25} {:>15}".format(
#                 layer,
#                 str(summary[layer]["output_shape"]),
#                 "{0:,}".format(summary[layer]["nb_params"]),
#             )

#         total_params += summary[layer]["nb_params"]
#         if show_input is True:
#             total_output += np.prod(summary[layer]["input_shape"])
#         else:
#             total_output += np.prod(summary[layer]["output_shape"])
#         if "trainable" in summary[layer]:
#             if summary[layer]["trainable"] == True:
#                 trainable_params += summary[layer]["nb_params"]

#         print(line_new)

#     print("=======================================================================")
#     print(f"Total params: {total_params:0,}")
#     print(f"Trainable params: {trainable_params:0,}")
#     print(f"Non-trainable params: {(total_params - trainable_params):0,}")
#     print("-----------------------------------------------------------------------")

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