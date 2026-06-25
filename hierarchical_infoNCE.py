from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


def get_ancestors(label, hierarchy):
    """
    Returns the set of ancestors for a given label.
    """
    ancestors = set()
    current_label = label
    while current_label is not None:
        ancestors.add(current_label)
        current_label = hierarchy.get(current_label)
    return ancestors


def hierarchical_similarity(label1, label2, hierarchy):
    """
    Computes the hierarchical similarity between two labels based on shared ancestors.
    """
    ancestors1 = get_ancestors(label1, hierarchy)
    ancestors2 = get_ancestors(label2, hierarchy)
    shared_ancestors = ancestors1.intersection(ancestors2)
    # Similarity can be defined as the number of shared ancestors
    similarity = len(shared_ancestors)
    return similarity


def hierarchical_similarity_between_label_sets(labels1, labels2, hierarchy):
    """
    Computes the hierarchical similarity between two sets of labels.
    """
    max_similarity = 0
    for label1 in labels1:
        for label2 in labels2:
            sim = hierarchical_similarity(label1, label2, hierarchy)
            if sim > max_similarity:
                max_similarity = sim
    return max_similarity


# def hierarchical_multi_positive_contrastive_loss(
#     anchors, positives_list, negatives, negative_weights, temperature=0.07
# ):
#     """
#     Computes the hierarchical InfoNCE loss with multiple positive examples per anchor.

#     Parameters:
#     - anchors: Tensor of shape [batch_size, embedding_dim], anchor embeddings.
#     - positives_list: List of Tensors, each of shape [num_positives_i, embedding_dim], positive embeddings for each anchor.
#     - negatives: Tensor of shape [num_negatives, embedding_dim], negative embeddings.
#     - negative_weights: Tensor of shape [batch_size, num_negatives], weights for each anchor-negative pair.
#     - temperature: Float, temperature parameter for scaling.

#     Returns:
#     - loss: Scalar tensor representing the hierarchical contrastive loss.
#     """
#     batch_size = anchors.size(0)
#     loss = 0.0

#     # Normalize embeddings
#     anchors_norm = F.normalize(anchors, p=2, dim=1)          # [batch_size, embedding_dim]
#     negatives_norm = F.normalize(negatives, p=2, dim=1)      # [num_negatives, embedding_dim]

#     for i in range(batch_size):
#         anchor = anchors_norm[i]                             # [embedding_dim]
#         positives = positives_list[i]                        # [num_positives_i, embedding_dim]
#         num_positives = positives.size(0)

#         if num_positives == 0:
#             continue  # Skip if no positive examples

#         # Normalize positives
#         positives_norm = F.normalize(positives, p=2, dim=1)  # [num_positives_i, embedding_dim]

#         # Compute similarities
#         pos_sim = torch.matmul(positives_norm, anchor) / temperature    # [num_positives_i]
#         neg_sim = torch.matmul(negatives_norm, anchor) / temperature    # [num_negatives]

#         # Apply exponential to similarities
#         exp_pos_sim = torch.exp(pos_sim)                                # [num_positives_i]
#         exp_neg_sim = torch.exp(neg_sim) * negative_weights[i]          # [num_negatives]

#         # Denominator includes both positives and negatives
#         denom = exp_pos_sim.sum() + exp_neg_sim.sum()

#         # Loss per positive
#         loss_i = - (1 / num_positives) * torch.sum(torch.log(exp_pos_sim / denom))

#         loss += loss_i

#     loss = loss / batch_size
#     return loss

def compute_negative_weights(anchor_labels_list, negative_labels_list, hierarchy, batch_size, device='cuda'):
    """
    Computes hierarchical weights between each anchor and all negatives.

    Parameters:
    - anchor_labels_list: List of label sets for each anchor (batch_size).
    - negative_labels_list: List of paths sets for each negative (num_negatives).

    Returns:
    - negative_weights: List, weights for each anchor-negative pair.
    """
    negative_weights_list = []

    for i in range(batch_size):
        anchor_labels = set(anchor_labels_list[i])  # Set of labels for anchor i
        negatives_labels_i = negative_labels_list[i]  # List of labels or paths for negatives

        negative_weights_i = []

        for neg_labels in negatives_labels_i:
            if isinstance(neg_labels, list):  # If neg_labels is a path
                neg_labels_set = set(neg_labels)
            else:  # If neg_labels is a set of labels
                neg_labels_set = set(neg_labels)

            sim = hierarchical_similarity_between_label_sets(anchor_labels, neg_labels_set, hierarchy)
            # Convert similarity to weight (e.g., exponential decay)
            weight = math.exp(-sim)
            negative_weights_i.append(weight)

        negative_weights_i = torch.tensor(negative_weights_i).to(device)
        negative_weights_list.append(negative_weights_i)

    return negative_weights_list


# def hierarchical_multi_positive_contrastive_loss(
#     anchors, positives_list, negatives_list, negative_weights_list, temperature=0.07
# ):
#     batch_size = anchors.size(0)
#     loss = 0.0
#     epsilon = 1e-8

#     # Normalize anchor embeddings
#     anchors_norm = F.normalize(anchors, p=2, dim=1)  # [batch_size, embedding_dim]

#     for i in range(batch_size):
#         anchor = anchors_norm[i]  # [embedding_dim]
#         positives = positives_list[i]  # List of positive embeddings for anchor i
#         negatives = negatives_list[i]  # Tensor of negatives for anchor i
#         negative_weights = negative_weights_list[i]  # Tensor of negative weights for anchor i

#         if len(positives) == 0 or negatives.size(0) == 0:
#             continue  # Skip if no positives or negatives

#         # Convert list of positive embeddings to tensor
#         positives = torch.stack(positives)  # [num_positives_i, embedding_dim]
#         # Normalize positives and negatives
#         positives_norm = F.normalize(positives, p=2, dim=1)  # [num_positives_i, embedding_dim]
#         negatives_norm = F.normalize(negatives, p=2, dim=1)  # [num_negatives_i, embedding_dim]

#         # Compute similarities
#         pos_sim = torch.matmul(positives_norm, anchor) / temperature  # [num_positives_i]
#         neg_sim = torch.matmul(negatives_norm, anchor) / temperature  # [num_negatives_i]
#         # print(f"pos_sim: {pos_sim}")
#         # print(f"neg_sim: {neg_sim}")

#         # Apply exponential to similarities
#         exp_pos_sim = torch.exp(pos_sim)  # [num_positives_i]
#         exp_neg_sim = torch.exp(neg_sim) * negative_weights  # [num_negatives_i]

#         # Denominator includes both positives and negatives
#         denom = exp_pos_sim.sum() + exp_neg_sim.sum() + epsilon

#         # if denom.item() == 0 or torch.isnan(denom):
#         #     print(f"Denominator is zero or NaN at anchor {i}.")
#         #     print(f"exp_pos_sim.sum(): {exp_pos_sim.sum().item()}, exp_neg_sim.sum(): {exp_neg_sim.sum().item()}")
#         #     continue  # Skip this anchor or handle appropriately

#         # Loss per positive
#         loss_i = - (1 / len(positives)) * torch.sum(torch.log(exp_pos_sim / denom))

#         loss += loss_i

#     loss = loss / batch_size
#     return loss

def hierarchical_info_nce_loss(
        anchors, positives_list, negatives_list, negatives_labels_list, labels_list, hierarchy, temperature,
        epsilon=1e-8
):
    batch_size = anchors.size(0)
    loss = 0.0

    # Normalize anchor embeddings
    anchors_norm = F.normalize(anchors, p=2, dim=1)  # [batch_size, embedding_dim]

    for i in range(batch_size):
        anchor = anchors_norm[i]  # [embedding_dim]
        positives = positives_list[i]  # List of positive embeddings for anchor i
        negatives = negatives_list[i]  # Tensor of negatives for anchor i
        negative_labels_i = negatives_labels_list[i]  # List of label sets for negatives
        anchor_labels = labels_list[i]  # Labels of anchor i

        if len(positives) == 0 or negatives.size(0) == 0:
            continue  # Skip if no positives or negatives

        # Normalize positives and negatives
        positives_norm = F.normalize(torch.stack(positives), p=2, dim=1)  # [num_positives_i, embedding_dim]
        negatives_norm = F.normalize(negatives, p=2, dim=1)  # [num_negatives_i, embedding_dim]

        # Compute similarities
        pos_sim = torch.matmul(positives_norm, anchor) / temperature  # [num_positives_i]
        neg_sim = torch.matmul(negatives_norm, anchor) / temperature  # [num_negatives_i]

        # # Compute hierarchical similarities for negatives
        # sim_list = []
        # for neg_labels in negative_labels_i:
        #     if isinstance(neg_labels[0], str):
        #         # If neg_labels is a path (list of labels), convert to set
        #         neg_labels_set = set(neg_labels)
        #     else:
        #         # If neg_labels is a set of labels
        #         neg_labels_set = set(neg_labels)
        #     sim = compute_hierarchical_similarity(anchor_labels, neg_labels_set, hierarchy)
        #     sim_list.append(sim)
        # sim_tensor = torch.tensor(sim_list).to(anchor.device)  # [num_negatives_i]

        # # Normalize hierarchical similarities to [0, 1]
        # max_sim_value = get_max_similarity(hierarchy)
        # sim_normalized = sim_tensor / max_sim_value

        # # Compute negative weights
        # negative_weights = torch.exp(-sim_normalized) + epsilon  # [num_negatives_i]

        # Use log-sum-exp for numerical stability
        max_sim = torch.max(torch.cat([pos_sim, neg_sim]))
        exp_pos_sim = torch.exp(pos_sim - max_sim)  # [num_positives_i]
        # exp_neg_sim = torch.exp(neg_sim - max_sim) * negative_weights  # [num_negatives_i]
        exp_neg_sim = torch.exp(neg_sim - max_sim)  # [num_negatives_i]

        denom = torch.log(torch.sum(exp_pos_sim) + torch.sum(exp_neg_sim) + epsilon) + max_sim  # Scalar
        log_prob = pos_sim - denom  # [num_positives_i]

        # Loss per positive
        loss_i = -torch.mean(log_prob)

        loss += loss_i

    loss = loss / batch_size
    return loss


def compute_hierarchical_similarity(anchor_labels, neg_labels, hierarchy):
    """
    Computes the hierarchical similarity between two sets of labels based on the number of common ancestors.

    Parameters:
    - anchor_labels: Set of labels for the anchor.
    - neg_labels: Set of labels for the negative sample.
    - hierarchy: The label hierarchy.

    Returns:
    - sim: Float, the number of common ancestors shared between the anchor and negative labels.
    """
    max_depth = get_max_depth(hierarchy)
    total_common_ancestors = 0

    for label_a in anchor_labels:
        ancestors_a = get_ancestors(label_a, hierarchy)
        for label_n in neg_labels:
            ancestors_n = get_ancestors(label_n, hierarchy)
            common_ancestors = ancestors_a.intersection(ancestors_n)
            total_common_ancestors += len(common_ancestors)

    # Normalize similarity
    sim = total_common_ancestors / (len(anchor_labels) * len(neg_labels) * max_depth)
    return sim


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


def get_max_depth(hierarchy):
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


def get_max_similarity(hierarchy):
    """
    Computes the maximum possible hierarchical similarity.

    Parameters:
    - hierarchy: The label hierarchy.

    Returns:
    - max_sim_value: Float, maximum possible similarity value.
    """
    max_depth = get_max_depth(hierarchy)
    # Maximum similarity occurs when all labels are the same and share all ancestors
    return 1.0  # Since sim is normalized to [0, 1]


class HierarchyContrastiveLoss(nn.Module):
    """
    Hierarchy-aware text ↔ path contrastive loss.

    Args:
        w_matrix: [P, P] tensor with w(i, j) = hierarchy-based similarity
                  between path i and path j (e.g. exp(-λ * distance(i, j))).
                  Row/col order must match the order of path_embs used later.
        tau:      temperature for the softmax over path similarities.
        reduction: 'mean' (default) or 'sum' over the batch.
    """

    def __init__(self, w_matrix, tau=0.3, reduction='mean'):
        super(HierarchyContrastiveLoss, self).__init__()

        if w_matrix.dim() != 2 or w_matrix.size(0) != w_matrix.size(1):
            raise ValueError("w_matrix must be a square [P, P] tensor.")

        # store as buffer so it moves with .to(device) / .cuda()
        self.register_buffer("w_matrix", w_matrix.float())
        self.tau = float(tau)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def _build_targets(self, batch_paths, device):
        """
        Build hierarchical soft target distribution q for each example.

        batch_paths: list (len B) of lists of path IDs (ints) for that example.
        Returns:
            q: [B, P] tensor with q[b, j] = probability mass for path j.
        """
        B = len(batch_paths)
        P = self.w_matrix.size(0)

        q = torch.zeros(B, P, device=device)

        for b, paths in enumerate(batch_paths):
            if len(paths) == 0:
                raise ValueError("Each example must have at least one path ID.")

            # w_rows: [k, P] where k = number of paths for this example
            idx = torch.as_tensor(paths, dtype=torch.long, device=device)
            w_rows = self.w_matrix.index_select(0, idx)  # [k, P]
            w_rows = F.normalize(w_rows, p=1, dim=1)
            w_sum = w_rows.sum(dim=0)  # [P]

            # Fallback if all zeros (should not happen, but be safe)
            if torch.all(w_sum == 0):
                w_sum = torch.zeros_like(w_sum)
                w_sum[idx] = 1.0

            q[b] = w_sum / w_sum.sum()

        return q  # [B, P]

    def forward(self, text_embs, path_embs, batch_paths):
        """
        Compute hierarchy-aware contrastive loss for a batch.

        Args:
            text_embs: [B, D] tensor of text embeddings (anchors).
            path_embs: [P, D] tensor of path embeddings. Order of P must match
                       the order used to build w_matrix.
            batch_paths: list of length B; each element is a list of path IDs
                         (ints) that are ground-truth for that example.

        Returns:
            scalar loss (if reduction != 'none') or [B] tensor of per-example loss.
        """
        if text_embs.dim() != 2 or path_embs.dim() != 2:
            raise ValueError("text_embs and path_embs must be 2D [*, D] tensors.")

        B, D = text_embs.size()
        P, Dp = path_embs.size()
        if D != Dp:
            raise ValueError("Embedding dimensions of text_embs and path_embs must match.")
        if len(batch_paths) != B:
            raise ValueError("batch_paths length must equal batch size B.")
        if P != self.w_matrix.size(0):
            raise ValueError("path_embs first dim must match w_matrix size.")

        device = text_embs.device
        path_embs = path_embs.to(device)

        # L2-normalize embeddings
        text_embs = F.normalize(text_embs, dim=-1)
        path_embs = F.normalize(path_embs, dim=-1)

        # [B, P] similarity logits divided by temperature
        logits = (text_embs @ path_embs.t()) / self.tau  # [B, P]
        log_p = F.log_softmax(logits, dim=-1)  # log p_j

        # Build hierarchical soft targets q_j
        q = self._build_targets(batch_paths, device=device)  # [B, P]
        entropy = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean()
        print(f"Entropy 1: {entropy.item()}")
        # Target sharpening (improves gradient signal)
        alpha = 1.5  # try 2.0-4.0
        q = q.pow(alpha)
        q = q / q.sum(dim=-1, keepdim=True)
        entropy = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean()
        print(f"Entropy sharpening: {entropy.item()}")

        with torch.no_grad():
            p = log_p.exp()
            ent_p = -(p * log_p).sum(dim=-1).mean()
            ent_q = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean()
            kl_qp = (q * (torch.log(q + 1e-8) - log_p)).sum(dim=-1).mean()
            print(f"ent_q={ent_q.item():.3f} ent_p={ent_p.item():.3f} kl(q||p)={kl_qp.item():.3f}")

        # Per-example loss: - Σ_j q_j log p_j
        loss_per_example = -(q * log_p).sum(dim=-1)  # [B]

        if self.reduction == "mean":
            return loss_per_example.mean()
        elif self.reduction == "sum":
            return loss_per_example.sum()
        else:  # 'none'
            return loss_per_example


from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class InBatchHierarchyContrastiveLoss(nn.Module):
    """
    Path-level in-batch hierarchical contrastive loss.

    Instead of contrasting each text embedding against all P paths, we build a
    candidate set C from the union of GT paths in the current batch. This yields
    stable in-batch negatives (other samples' GT paths) and is typically much
    easier to optimize early in training.

    Args:
        w_matrix: [P, P] hierarchy similarity matrix aligned with global path ids.
        tau: temperature for softmax over candidate similarities.
        alpha: target sharpening power; alpha=1.0 disables sharpening.
        reduction: 'mean' (default), 'sum', or 'none' over valid samples.
    """

    def __init__(
            self,
            w_matrix: torch.Tensor,
            tau: float = 0.3,
            alpha: float = 2.0,
            gamma: float = 0.3,
            reduction: str = "mean",
    ):
        super().__init__()
        if w_matrix.dim() != 2 or w_matrix.size(0) != w_matrix.size(1):
            raise ValueError("w_matrix must be a square [P, P] tensor.")
        self.register_buffer("w_matrix", w_matrix.float())
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    @staticmethod
    def _unique_in_order(ids: Sequence[int]) -> list[int]:
        seen = set()
        out: list[int] = []
        for x in ids:
            xi = int(x)
            if xi not in seen:
                seen.add(xi)
                out.append(xi)
        return out

    def forward(self, text_embs: torch.Tensor, path_embs: torch.Tensor, batch_paths: Sequence[Sequence[int]]):
        """
        Args:
            text_embs: [B, D] text embeddings (anchors).
            path_embs: [P, D] global path embedding table (aligned with w_matrix).
            batch_paths: length B; each element is a list of GT path ids for that sample.

        Returns:
            scalar loss (or [B_valid] if reduction='none') computed over valid samples.
        """
        if text_embs.dim() != 2 or path_embs.dim() != 2:
            raise ValueError("text_embs and path_embs must be 2D [*, D] tensors.")
        B, D = text_embs.shape
        P, Dp = path_embs.shape
        if D != Dp:
            raise ValueError("Embedding dims of text_embs and path_embs must match.")
        if len(batch_paths) != B:
            raise ValueError("batch_paths length must equal batch size B.")
        if P != self.w_matrix.size(0):
            raise ValueError("path_embs first dim must match w_matrix size.")

        device = text_embs.device

        # ---- Build candidate set C from GT paths in batch ----
        flat: list[int] = []
        for gt in batch_paths:
            if gt is None:
                continue
            flat.extend(list(gt))

        candidates = self._unique_in_order([x for x in flat if x is not None])
        if len(candidates) == 0:
            return torch.zeros((), device=device, dtype=text_embs.dtype)

        cand = torch.tensor(candidates, device=device, dtype=torch.long)  # [C]
        cand_path_embs = path_embs.index_select(0, cand)  # [C, D]

        # Normalize embeddings for cosine similarity
        t = F.normalize(text_embs, dim=-1)
        p = F.normalize(cand_path_embs, dim=-1)

        # logits over candidates: [B, C]
        logits = (t @ p.t()) / self.tau
        log_p = F.log_softmax(logits, dim=-1)

        # ---- Build soft targets q over candidates ----
        q = torch.zeros((B, cand.numel()), device=device, dtype=text_embs.dtype)
        valid = torch.zeros((B,), device=device, dtype=torch.bool)

        for b in range(B):
            gt = batch_paths[b]
            if gt is None or len(gt) == 0:
                continue
            valid[b] = True

            gt_idx = torch.as_tensor(gt, dtype=torch.long, device=device)  # [k]
            # w_rows: [k, C]
            w_rows = self.w_matrix.index_select(0, gt_idx).index_select(1, cand)
            # L1 normalize each row to remove scale; then sum across GT paths
            w_rows = F.normalize(w_rows, p=1, dim=1)
            # w_sum = w_rows.sum(dim=0)  # [C]
            # w_sum = w_sum / (w_sum.sum() + 1e-12)
            # q[b] = w_sum
            w_sum = w_rows.sum(dim=0)
            w_sum = w_sum / (w_sum.sum() + 1e-12)

            # ----- exact GT one-hot over current candidate set -----
            one_hot = torch.zeros_like(w_sum)
            gt_local = []

            for gid in gt_idx.tolist():
                pos = (cand == gid).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    gt_local.append(pos.item())

            if len(gt_local) > 0:
                for pos in gt_local:
                    one_hot[pos] = 1.0 / len(gt_local)

            # gamma controls how much exact-GT mass we inject
            q[b] = (1.0 - self.gamma) * w_sum + self.gamma * one_hot

        if valid.sum().item() == 0:
            return torch.zeros((), device=device, dtype=text_embs.dtype)

        # target sharpening (controls entropy)
        if self.alpha is not None and self.alpha != 1.0:
            q = q.pow(self.alpha)
            q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)

        # with torch.no_grad():
        #     if B > 0:
        #         b0 = 0
        #         p0 = log_p[b0].exp()
        #
        #         top_pred_val, top_pred_idx = torch.topk(p0, k=min(5, p0.numel()))
        #         top_q_val, top_q_idx = torch.topk(q[b0], k=min(5, q[b0].numel()))
        #
        #         print("\n===== HCL SAMPLE DEBUG =====")
        #         print("gt path ids:", batch_paths[b0])
        #         print("candidate size:", cand.numel())
        #         print("top pred path ids:", cand[top_pred_idx].tolist())
        #         print("top pred probs:", top_pred_val.tolist())
        #         print("top q path ids:", cand[top_q_idx].tolist())
        #         print("top q probs:", top_q_val.tolist())
        #
        #         gt_set = set(batch_paths[b0])
        #         top_q_set = set(cand[top_q_idx].tolist())
        #         top_pred_set = set(cand[top_pred_idx].tolist())
        #         print("GT in top-q?:", len(gt_set & top_q_set) > 0)
        #         print("GT in top-pred?:", len(gt_set & top_pred_set) > 0)

        # with torch.no_grad():
        #     p = log_p.exp()
        #     ent_p = -(p * log_p).sum(dim=-1).mean()
        #     ent_q = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean()
        #     kl_qp = (q * (torch.log(q + 1e-8) - log_p)).sum(dim=-1).mean()
        #     print(f"ent_q={ent_q.item():.3f} ent_p={ent_p.item():.3f} kl(q||p)={kl_qp.item():.3f}")
        #     C = cand.numel()
        #     logC = float(torch.log(torch.tensor(C, device=device)))
        #     print(f"C={C} logC={logC:.3f} ent_p={ent_p.item():.3f} ent_q={ent_q.item():.3f} kl={kl_qp.item():.3f}")

        # ---- Loss: cross-entropy with soft targets ----
        loss_per = -(q * log_p).sum(dim=-1)  # [B]
        loss_per = loss_per[valid]

        if self.reduction == "mean":
            return loss_per.mean()
        elif self.reduction == "sum":
            return loss_per.sum()
        else:  # 'none'
            return loss_per


class NewInBatchHierarchyContrastiveLoss(nn.Module):
    """
    Hierarchy-aware contrastive loss.

    Two modes:
    1) Old mode (backward compatible):
       candidate_ids_per_sample is None
       -> use the union of GT paths in the current batch as candidate set
    2) New mode:
       candidate_ids_per_sample is provided
       -> each sample uses its own candidate set (GT + sampled negatives)
    """

    def __init__(
            self,
            w_matrix: torch.Tensor,
            tau: float = 0.3,
            alpha: float = 2.0,
            gamma: float = 0.3,
            reduction: str = "mean",
    ):
        super().__init__()
        if w_matrix.dim() != 2 or w_matrix.size(0) != w_matrix.size(1):
            raise ValueError("w_matrix must be a square [P, P] tensor.")
        self.register_buffer("w_matrix", w_matrix.float())
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    @staticmethod
    def _unique_in_order(ids: Sequence[int]) -> list[int]:
        seen = set()
        out: list[int] = []
        for x in ids:
            xi = int(x)
            if xi not in seen:
                seen.add(xi)
                out.append(xi)
        return out

    def _build_q_for_candidates(
            self,
            gt_ids: Sequence[int],
            cand_ids: Sequence[int],
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build soft target q over a provided candidate set.
        """
        gt_ids = self._unique_in_order(gt_ids)
        cand_ids = self._unique_in_order(cand_ids)

        gt_idx = torch.as_tensor(gt_ids, dtype=torch.long, device=device)
        cand = torch.as_tensor(cand_ids, dtype=torch.long, device=device)

        # [num_gt, C]
        w_rows = self.w_matrix.index_select(0, gt_idx).index_select(1, cand)
        w_rows = F.normalize(w_rows, p=1, dim=1)

        # aggregate GT rows -> [C]
        w_sum = w_rows.sum(dim=0)
        w_sum = w_sum / (w_sum.sum() + 1e-12)

        # exact GT one-hot over current candidate set
        one_hot = torch.zeros_like(w_sum, dtype=dtype, device=device)
        gt_local = []
        for gid in gt_idx.tolist():
            pos = (cand == gid).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                gt_local.append(pos.item())

        if len(gt_local) > 0:
            for pos in gt_local:
                one_hot[pos] = 1.0 / len(gt_local)

        q = (1.0 - self.gamma) * w_sum + self.gamma * one_hot

        if self.alpha is not None and self.alpha != 1.0:
            q = q.pow(self.alpha)
            q = q / (q.sum(dim=-1, keepdim=False) + 1e-12)

        return q

    def forward(
            self,
            text_embs: torch.Tensor,
            path_embs: torch.Tensor,
            batch_paths: Sequence[Sequence[int]],
            candidate_ids_per_sample: Sequence[Sequence[int]] | None = None,
    ):
        """
        Args:
            text_embs: [B, D]
            path_embs: [P, D]
            batch_paths: GT path ids for each sample
            candidate_ids_per_sample:
                None -> old in-batch mode
                provided -> new sample-specific candidate mode
        """
        if text_embs.dim() != 2 or path_embs.dim() != 2:
            raise ValueError("text_embs and path_embs must be 2D [*, D] tensors.")
        B, D = text_embs.shape
        P, Dp = path_embs.shape
        if D != Dp:
            raise ValueError("Embedding dims of text_embs and path_embs must match.")
        if len(batch_paths) != B:
            raise ValueError("batch_paths length must equal batch size B.")
        if P != self.w_matrix.size(0):
            raise ValueError("path_embs first dim must match w_matrix size.")
        if candidate_ids_per_sample is not None and len(candidate_ids_per_sample) != B:
            raise ValueError("candidate_ids_per_sample length must equal batch size B.")

        device = text_embs.device

        # Normalize embeddings once
        t_all = F.normalize(text_embs, dim=-1)
        p_all = F.normalize(path_embs, dim=-1)

        # ------------------------------------------------------------------
        # Mode A: old behavior (shared in-batch candidate set)
        # ------------------------------------------------------------------
        if candidate_ids_per_sample is None:
            flat: list[int] = []
            for gt in batch_paths:
                if gt is None:
                    continue
                flat.extend(list(gt))

            candidates = self._unique_in_order([x for x in flat if x is not None])
            if len(candidates) == 0:
                return torch.zeros((), device=device, dtype=text_embs.dtype)

            cand = torch.tensor(candidates, device=device, dtype=torch.long)  # [C]
            cand_path_embs = p_all.index_select(0, cand)  # [C, D]

            logits = (t_all @ cand_path_embs.t()) / self.tau  # [B, C]
            log_p = F.log_softmax(logits, dim=-1)

            q = torch.zeros((B, cand.numel()), device=device, dtype=text_embs.dtype)
            valid = torch.zeros((B,), device=device, dtype=torch.bool)

            for b in range(B):
                gt = batch_paths[b]
                if gt is None or len(gt) == 0:
                    continue
                valid[b] = True
                q[b] = self._build_q_for_candidates(
                    gt_ids=gt,
                    cand_ids=candidates,
                    device=device,
                    dtype=text_embs.dtype,
                )

            if valid.sum().item() == 0:
                return torch.zeros((), device=device, dtype=text_embs.dtype)

            loss_per = -(q * log_p).sum(dim=-1)  # [B]
            loss_per = loss_per[valid]

            if self.reduction == "mean":
                return loss_per.mean()
            elif self.reduction == "sum":
                return loss_per.sum()
            else:
                return loss_per

        # ------------------------------------------------------------------
        # Mode B: new behavior (sample-specific candidate set)
        # ------------------------------------------------------------------
        loss_terms = []

        for b in range(B):
            gt = batch_paths[b]
            if gt is None or len(gt) == 0:
                continue

            # ensure GT always appears in candidate set
            cand_ids = self._unique_in_order(list(gt) + list(candidate_ids_per_sample[b]))
            if len(cand_ids) == 0:
                continue

            cand = torch.tensor(cand_ids, device=device, dtype=torch.long)
            cand_path_embs = p_all.index_select(0, cand)  # [C_b, D]

            logits = (t_all[b:b + 1] @ cand_path_embs.t()) / self.tau  # [1, C_b]
            log_p = F.log_softmax(logits, dim=-1).squeeze(0)  # [C_b]

            q = self._build_q_for_candidates(
                gt_ids=gt,
                cand_ids=cand_ids,
                device=device,
                dtype=text_embs.dtype,
            )  # [C_b]

            loss_b = -(q * log_p).sum()
            loss_terms.append(loss_b)

        if len(loss_terms) == 0:
            return torch.zeros((), device=device, dtype=text_embs.dtype)

        loss_per = torch.stack(loss_terms)

        if self.reduction == "mean":
            return loss_per.mean()
        elif self.reduction == "sum":
            return loss_per.sum()
        else:
            return loss_per