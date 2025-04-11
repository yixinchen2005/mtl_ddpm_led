import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

class CrossEntropyLossWrap(nn.Module):
    def __init__(self, num_labels, option="standard"):
        super().__init__()
        self.num_labels = num_labels
        self.option = option
        if self.option == "standard":
            self.loss = CrossEntropyLoss(ignore_index=0, reduction='sum')
        else:
            self.loss = CrossEntropyLoss(ignore_index=0, reduction='none')

    def forward(self, emissions, targets, masks=None, weights=None):
        bsz = targets.shape[0]
        emissions = emissions.view(-1, self.num_labels)
        targets = targets.view(-1)
        out = self.loss(emissions, targets)
        if self.option == "standard":
            return out
        elif self.option == "per_sample": 
            out = out.view(bsz, -1)
            return out.sum(dim=-1)
        elif self.option == "weighted":
            out = out.view(bsz, -1)
            weights = weights.view(-1, 1)
            out = out * masks.float() * weights
            return out.sum()
        else:
            return 0
    
class ContrastiveLoss(nn.Module):
    def __init__(self, option="dist", margin=1.0, temperature=0.07):
        super().__init__()
        self.option = option
        if option == "dist":
            self.margin = margin
        elif option == "infonce":
            self.temperature = temperature
    
    def dist_loss(self, features_old, features_new):
        """
        Contrastive loss to maximize the difference between feature embeddings of targets_old and targets_new.
        """
        distance = F.cosine_similarity(features_old, features_new, dim=-1)
        loss = torch.mean(F.relu(self.margin - distance)) # Push apart
        return loss
    
    def info_nce_loss(self, features_old, features_new):
        """
        InfoNCE loss to push targets_old and targets_new apart while pulling similar embeddings together.
        """
        features = torch.cat([features_old, features_new], dim=0)  # Stack all features
        similarity_matrix = torch.mm(features, features.T)  # Compute pairwise similarity

        # Get positive pairs (diagonal) and negatives (off-diagonal)
        batch_size = features_old.shape[0]
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(features.device)
        
        # Compute NT-Xent loss
        logits = similarity_matrix / self.temperature
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(self, features_old, features_new):
        out = 0
        if self.option == "dist":
            out = self.dist_loss(features_old, features_new)
        elif self.option == "infonce":
            out = self.info_nce_loss(features_old, features_new)
        return out
    
class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emissions, teacher_emissions):
        """
        Computes Mean Teacher consistency loss using KL divergence for sequence data.
        - student_logits: (bsz, len, num_classes)
        - teacher_logits: (bsz, len, num_classes)
        - temperature: Smoothing factor to control the sharpness of teacher predictions
        """
        teacher_probs = F.softmax(teacher_emissions / self.temperature, dim=-1)  # Soft probabilities over classes
        student_log_probs = F.log_softmax(student_emissions, dim=-1)  # Log probabilities over classes

        # KL divergence applied per token, then averaged over batch and sequence length
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        return loss
    
class BCEWithLogitsLossWrap(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.loss = BCEWithLogitsLoss(reduction='sum')

    def forward(self, emissions, targets):
        out = self.loss(emissions, targets)
        return out
    
class MSELossWrap(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.loss = MSELoss(reduction='none')

    def forward(self, emissions, targets, attention_mask):
        #targets = F.one_hot(targets, self.num_labels).to(torch.float32)
        assert emissions.shape == targets.shape
        loss = self.loss(emissions, targets)
        out = torch.sum(loss * attention_mask)
        return out
    
class FocalLoss(nn.Module):
    def __init__(self, num_labels, alpha=None, gamma=2.0, reduction='sum'):
        """
        Focal Loss for multi-class classification.

        :param alpha: Tensor of shape (num_classes,) for class weights. If None, no weighting is applied.
        :param gamma: Focusing parameter to down-weight easy examples.
        :param reduction: 'mean', 'sum', or 'none' for loss reduction.
        """
        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.alpha = alpha  # Class-wise weighting
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = CrossEntropyLoss(ignore_index=0, reduction='none')
        # self.ce_loss = CrossEntropyLoss(reduction='none')

    def forward(self, emissions, targets):
        """
        :param logits: Tensor of shape (batch_size, num_classes).
        :param targets: Tensor of shape (batch_size,) with class indices (not one-hot).
        """
        emissions = emissions.view(-1, self.num_labels)
        targets = targets.view(-1)
        ce_loss = self.ce_loss(emissions, targets)  # Standard Cross-Entropy Loss
        probs = torch.softmax(emissions, dim=-1)  # Convert logits to probabilities
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=emissions.shape[-1])  # One-hot labels
        pt = (probs * targets_one_hot).sum(dim=-1)  # Get probability of the true class
        focal_weight = (1 - pt) ** self.gamma  # Compute focal scaling factor
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets)
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction
        
class BinaryFocalLossWithMask(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, neg_weight=5.0, reduction="mean"):
        """
        Binary focal loss with attention mask support.
        - alpha: Weighting factor for class imbalance.
        - gamma: Modulation factor to focus on hard examples.
        - reduction: 'mean' (default) or 'sum' for aggregation.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # Keep per-token loss

    def forward(self, logits, targets, attention_mask=None):
        """
        Computes focal loss while ignoring padded tokens.
        - logits: Raw output from the discriminator (before sigmoid).
        - targets: Binary labels (1 for real, 0 for fake).
        - attention_mask: (Optional) 1 for valid tokens, 0 for padding.
        """
        bce_loss = self.bce(logits, targets)  # Compute standard BCE loss
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        pt = torch.where(targets == 1, probs, 1 - probs)  # Select p_t based on targets

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weight #
        alpha_weight = torch.where(targets == 0, self.alpha * self.neg_weight, (1 - self.alpha))

        # Apply focal weight to BCE loss
        loss = alpha_weight * focal_weight * bce_loss

        # Apply attention mask: ignore padding tokens
        if attention_mask is not None:
            loss = loss * attention_mask  # Zero out masked positions

        # Aggregate loss based on reduction mode
        if self.reduction == "mean":
            return loss.sum() / (attention_mask.sum() + 1e-8)  # Avoid division by zero
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction, return per-token loss