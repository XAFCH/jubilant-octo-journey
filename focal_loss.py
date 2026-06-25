import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalFocalLossWithPenalty(nn.Module):
    def __init__(self, hierarchy, gamma=2.0, alpha=None, lambda_hier=1.0, smoothing=0.1, reduction='mean'):
        super(HierarchicalFocalLossWithPenalty, self).__init__()
        self.hierarchy = hierarchy  # Dictionary mapping label indices to their parent indices
        self.gamma = gamma          # Focusing parameter for focal loss
        self.alpha = alpha          # Class weighting factor (can be a tensor of class weights)
        self.lambda_hier = lambda_hier  # Weight for the hierarchical penalty
        self.smoothing = smoothing  # Label smoothing factor
        self.reduction = reduction  # Reduction method ('mean' or 'sum')

    def forward(self, logits, targets):
        """
        logits: Tensor of shape [batch_size, num_labels], raw output from the model.
        targets: Tensor of shape [batch_size, num_labels], original binary target labels.
        """
        # Apply label smoothing
        targets_smoothed = self.label_smoothing(targets, self.smoothing)
        
        # Compute standard focal loss with smoothed labels
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smoothed, reduction='none'
        )
        p_t = torch.exp(-bce_loss)
        
        if self.alpha is not None:
            # If alpha is provided, adjust the loss
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        else:
            focal_loss = ((1 - p_t) ** self.gamma) * bce_loss

        # Compute hierarchical penalty
        hier_penalty = self.hierarchical_penalty(logits, targets)
        
        # Combine focal loss and hierarchical penalty
        total_loss = focal_loss + self.lambda_hier * hier_penalty
        
        if self.reduction == 'mean':
            return total_loss.mean()
        else:
            return total_loss.sum()

    def label_smoothing(self, targets, smoothing):
        """
        Applies label smoothing to the target labels.
        """
        # Replace 1s with (1 - smoothing) and 0s with smoothing
        targets_smoothed = targets * (1 - smoothing) + smoothing * (1 - targets)
        return targets_smoothed

    def hierarchical_penalty(self, logits, targets):
        """
        Computes the hierarchical penalty.
        """
        batch_size, num_labels = logits.shape
        sigmoid_logits = torch.sigmoid(logits)

        penalty = torch.zeros_like(sigmoid_logits)

        for child_label in range(num_labels):
            parent_label = self.hierarchy.get(child_label)
            if parent_label is not None:
                child_prob = sigmoid_logits[:, child_label]
                parent_prob = sigmoid_logits[:, parent_label]
                # Penalty when child is predicted but parent is not
                penalty[:, child_label] = child_prob * (1 - parent_prob)

        return penalty  # This will be added to the focal loss

