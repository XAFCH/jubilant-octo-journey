import torch
import torch.nn.functional as F

# compute label depths and max depth
def compute_label_depths(hierarchy):
    depths = {}
    def dfs(label, depth):
        depths[label] = depth
        children = [child for child, parent in hierarchy.items() if parent == label]
        for child in children:
            dfs(child, depth + 1)
    roots = [label for label, parent in hierarchy.items() if parent is None]
    for root in roots:
        dfs(root, 0)
    return depths, max(depths.values())

# get ancestors and find LCA
def get_ancestors(label, hierarchy):
    ancestors = []
    while label is not None:
        ancestors.append(label)
        label = hierarchy.get(label)
    return ancestors

def find_lca(label1, label2, hierarchy, label_depths):
    ancestors1 = set(get_ancestors(label1, hierarchy))
    ancestors2 = set(get_ancestors(label2, hierarchy))
    common_ancestors = ancestors1.intersection(ancestors2)
    if not common_ancestors:
        return None
    # Return the deepest common ancestor
    return max(common_ancestors, key=lambda x: label_depths[x])

# compute label similarity 
def compute_label_similarity(label1, label2, hierarchy, label_depths, max_depth):
    lca = find_lca(label1, label2, hierarchy, label_depths)
    if lca is None:
        return 0.0
    lca_depth = label_depths[lca]
    depth_diff = abs(label_depths[label1] - label_depths[label2])
    similarity = lca_depth / ((depth_diff + 1) * max_depth)
    return similarity

# compute similarity between label sets
def compute_label_set_similarity(labels1, labels2, hierarchy, label_depths, max_depth):
    total_similarity = 0.0
    count = 0
    for label1 in labels1:
        for label2 in labels2:
            sim = compute_label_similarity(label1, label2, hierarchy, label_depths, max_depth)
            total_similarity += sim
            count += 1
    if count == 0:
        return 0.0
    return total_similarity / count

def hierarchical_info_nce_loss(
    anchors, positives_list, negatives_list, labels_list, positives_labels_list, negatives_labels_list, hierarchy, temperature=0.07, epsilon=1e-8
):
    batch_size = anchors.size(0)
    loss = 0.0

    # Precompute label depths and max depth
    label_depths, max_depth = compute_label_depths(hierarchy)

    # Normalize anchor embeddings
    anchors_norm = F.normalize(anchors, p=2, dim=1)  # [batch_size, embedding_dim]

    for i in range(batch_size):
        anchor = anchors_norm[i]  # [embedding_dim]
        anchor_labels = labels_list[i]

        # Positives
        positives = positives_list[i]
        if len(positives) == 0:
            continue
        positives_norm = F.normalize(torch.stack(positives), p=2, dim=1)  # [num_positives_i, embedding_dim]
        pos_sims = torch.matmul(positives_norm.squeeze(1), anchor.unsqueeze(-1)).squeeze(-1) / temperature  # [num_positives_i]

        # Compute positive weights
        pos_weights_list = []
        for pos_labels in positives_labels_list[i]:
            sim = compute_label_set_similarity(anchor_labels, pos_labels, hierarchy, label_depths, max_depth)
            pos_weights_list.append(sim)
        pos_weights = torch.tensor(pos_weights_list, device=anchor.device, dtype=pos_sims.dtype).view(-1)  # [num_positives_i]

        # Adjust positive similarities by weights
        adjusted_pos_sims = pos_sims * pos_weights  # [num_positives_i]

        # Negatives
        negatives = negatives_list[i]
        if negatives.size(0) == 0:
            continue
        negatives_norm = F.normalize(negatives, p=2, dim=1)  # [num_negatives_i, embedding_dim]
        neg_sims = torch.matmul(negatives_norm.squeeze(1), anchor) / temperature  # [num_negatives_i]

        # Compute negative weights
        neg_weights_list = []
        for neg_labels in negatives_labels_list[i]:
            sim = compute_label_set_similarity(anchor_labels, neg_labels, hierarchy, label_depths, max_depth)
            neg_weights_list.append(1.0 - sim)
        neg_weights = torch.tensor(neg_weights_list, device=anchor.device, dtype=neg_sims.dtype).view(-1)  # [num_negatives_i]

        # Adjust negative similarities by weights
        adjusted_neg_sims = neg_sims * neg_weights  # [num_negatives_i]

        # Concatenate similarities
        all_sims = torch.cat([adjusted_pos_sims, adjusted_neg_sims], dim=0)  # [num_positives_i + num_negatives_i]

        # Compute loss
        max_sim = torch.max(all_sims).detach()
        exp_sims = torch.exp(all_sims - max_sim)
        denom = torch.log(torch.sum(exp_sims) + epsilon) + max_sim

        log_prob = adjusted_pos_sims - denom  # [num_positives_i]
        loss_i = -log_prob.mean()  # Scalar

        loss += loss_i

    loss = loss / batch_size
    return loss