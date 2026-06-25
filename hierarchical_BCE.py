import torch
import torch.nn as nn

def hierarchical_label_smoothing(labels, hierarchy, smoothing=0.2):
    """
    Applies hierarchical label smoothing to the target labels.

    Parameters:
    - labels: Tensor of shape [batch_size, num_labels], original binary target labels.
    - hierarchy: Dictionary mapping label indices to their parent indices.
    - smoothing: Float, smoothing value to assign to ancestor labels.

    Returns:
    - smoothed_labels: Tensor of shape [batch_size, num_labels], smoothed target labels.
    """
    batch_size, num_labels = labels.shape
    smoothed_labels = labels.clone()

    for i in range(batch_size):
        positive_indices = torch.where(labels[i] == 1)[0]  # Indices of positive labels
        visited = set()
        for pos_idx in positive_indices:
            current_label = pos_idx.item()
            # Traverse ancestors
            while current_label is not None and current_label in hierarchy:
                parent_label = hierarchy[current_label]
                if parent_label is not None and parent_label not in visited and labels[i, parent_label] == 0:
                    smoothed_labels[i, parent_label] = smoothing
                    visited.add(parent_label)
                current_label = parent_label
    return smoothed_labels

def hierarchical_penalty_loss(logits, hierarchy, lambda_hier=1.0):
    """
    Computes a hierarchical penalty loss.

    Parameters:
    - logits: Tensor of shape [batch_size, num_labels], raw output from the model (before sigmoid).
    - hierarchy: Dictionary mapping label indices to their parent indices.
    - lambda_hier: Float, weight for the hierarchical penalty.

    Returns:
    - penalty_loss: Scalar tensor representing the hierarchical penalty loss.
    """
    batch_size, num_labels = logits.shape
    sigmoid_logits = torch.sigmoid(logits)

    penalty = 0.0
    for i in range(batch_size):
        for child_label in range(num_labels):
            parent_label = hierarchy.get(child_label)
            if parent_label is not None:
                child_prob = sigmoid_logits[i, child_label]
                parent_prob = sigmoid_logits[i, parent_label]
                # Penalty when child is predicted but parent is not
                penalty += child_prob * (1 - parent_prob)
    penalty_loss = lambda_hier * penalty / batch_size
    return penalty_loss

def hierarchical_bce_loss_with_penalty(logits, labels, hierarchy, lambda_hier=1.0, smoothing=0.2):
    """
    Computes the BCE loss with hierarchical label smoothing and a hierarchical penalty.

    Parameters:
    - logits: Tensor of shape [batch_size, num_labels], raw output from the model (before sigmoid).
    - labels: Tensor of shape [batch_size, num_labels], original binary target labels.
    - hierarchy: Dictionary mapping label indices to their parent indices.
    - lambda_hier: Float, weight for the hierarchical penalty.
    - smoothing: Float, smoothing value for hierarchical label smoothing.

    Returns:
    - total_loss: Scalar tensor representing the total loss.
    """
    # Apply hierarchical label smoothing
    smoothed_labels = hierarchical_label_smoothing(labels, hierarchy, smoothing)

    # Compute BCE loss with smoothed labels
    bce_loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = bce_loss_fn(logits, smoothed_labels)

    # Compute hierarchical penalty loss
    penalty_loss = hierarchical_penalty_loss(logits, hierarchy, lambda_hier)

    # Total loss
    total_loss = bce_loss + penalty_loss
    return total_loss
