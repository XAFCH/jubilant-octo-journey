import torch
import torch.nn as nn
import torch.nn.functional as F
 
def get_ancestors(label, hierarchy):
    """
    Returns the set of ancestors for a given label.

    Parameters:
    - label: The label whose ancestors are to be found.
    - hierarchy: The label hierarchy.

    Returns:
    - ancestors: Set of ancestor labels.
    """
    ancestors = set()
    current_label = label
    while current_label is not None:
        ancestors.add(current_label)
        current_label = hierarchy.get(current_label)
    return ancestors

def compute_max_depth(hierarchy):
    """
    Computes the maximum depth of the hierarchy.

    Parameters:
    - hierarchy: The label hierarchy.

    Returns:
    - max_depth: Integer, maximum depth of the hierarchy.
    """
    max_depth = 0
    for label in hierarchy.keys():
        depth = len(get_ancestors(label, hierarchy))
        if depth > max_depth:
            max_depth = depth
    return max_depth

def get_label_path(label, hierarchy):
    """
    label: The label for which to get the path.
    hierarchy: A dictionary representing the hierarchy.
    
    Returns:
    path: A list representing the path from root to the label.
    """
    path = []
    while label is not None:
        path.insert(0, label)
        label = hierarchy.get(label)
    return path

# Remove Redundant Subpaths
def remove_subpaths(paths):
    """
    paths: List of label paths (each path is a list of label indices).

    Returns:
    filtered_paths: List of paths with subpaths removed.
    """
    # Sort paths by length in descending order
    paths_sorted = sorted(paths, key=lambda x: len(x), reverse=True)
    filtered_paths = []
    for path in paths_sorted:
        is_subpath = False
        for existing_path in filtered_paths:
            if is_subpath_of(path, existing_path):
                is_subpath = True
                break
        if not is_subpath:
            filtered_paths.append(path)
    return filtered_paths

def is_subpath_of(shorter, longer):
    """
    Checks if 'shorter' is a subpath of 'longer'.

    shorter, longer: Lists of label indices representing paths.

    Returns:
    True if 'shorter' is a subpath of 'longer'; False otherwise.
    """
    if len(shorter) >= len(longer):
        return False
    return shorter == longer[:len(shorter)]

# Compute Hierarchical Similarities
def compute_hierarchical_similarity(anchor_paths, negative_paths, max_depth):
    """
    Computes the hierarchical similarity between an anchor and negative sample.

    Parameters:
    - anchor_paths: List of label paths (after removing subpaths) for the anchor.
                     Each path is a list of label indices.
    - negative_paths: List of label paths (after removing subpaths) for the negative.
                      Each path is a list of label indices.
    - max_depth: The maximum depth of the hierarchy.

    Returns:
    - max_similarity: The maximum hierarchical similarity between any pair of paths.
                      A float value between 0 and 1.
    """
    max_similarity = 0.0
    for anchor_path in anchor_paths:
        for negative_path in negative_paths:
            num_shared = count_shared_ancestors(anchor_path, negative_path)
            sim_score = num_shared / max_depth
            if sim_score > max_similarity:
                max_similarity = sim_score
    return max_similarity

def count_shared_ancestors(path_a, path_b):
    """
    Counts the number of shared ancestors between two label paths.

    Parameters:
    - path_a: List of label indices representing a path from root to a label.
    - path_b: List of label indices representing another path from root to a label.

    Returns:
    - num_shared: Integer count of the number of shared ancestors.
    """
    num_shared = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            num_shared += 1
        else:
            break
    return num_shared

import torch
import torch.nn.functional as F

def hierarchical_contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings, weights, temperature=0.07):
    """
    Computes the hierarchical contrastive loss for a batch of samples.

    Parameters:
    - anchor_embeddings: Tensor of shape [batch_size, embedding_dim], embeddings of anchor samples.
    - positive_embeddings: Tensor of shape [batch_size, embedding_dim], embeddings of positive samples.
    - negative_embeddings: Tensor of shape [batch_size, num_negatives, embedding_dim], embeddings of negative samples.
    - weights: Tensor of shape [batch_size, num_negatives], containing hierarchical similarity scores (weights).
    - temperature: Float, temperature parameter for scaling similarities.

    Returns:
    - loss: Scalar tensor representing the computed contrastive loss.
    """
    # Normalize embeddings
    anchor_norm = F.normalize(anchor_embeddings, dim=1)  # [batch_size, embedding_dim]
    positive_norm = F.normalize(positive_embeddings, dim=1)  # [batch_size, embedding_dim]
    negative_norm = F.normalize(negative_embeddings, dim=2)  # [batch_size, num_negatives, embedding_dim]

    # Compute positive similarities
    pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)  # [batch_size]
    pos_sim = pos_sim / temperature

    # Compute negative similarities
    anchor_norm_expanded = anchor_norm.unsqueeze(1)  # [batch_size, 1, embedding_dim]
    neg_sims = torch.sum(anchor_norm_expanded * negative_norm, dim=2)  # [batch_size, num_negatives]
    neg_sims = neg_sims / temperature

    # Adjust negative similarities using hierarchical weights
    # Since higher weights indicate higher similarity (more shared ancestors), we reduce the impact of such negatives
    adjusted_neg_sims = neg_sims * (1 - weights)  # Reduce the influence of similar negatives

    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim.unsqueeze(1), adjusted_neg_sims], dim=1)  # [batch_size, 1 + num_negatives]

    # Labels: zeros since positives are at index 0
    labels = torch.zeros(anchor_embeddings.size(0), dtype=torch.long).to(anchor_embeddings.device)

    # Compute loss
    loss = F.cross_entropy(logits, labels)
    return loss
