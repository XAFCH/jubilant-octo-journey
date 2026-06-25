from __future__ import annotations
import os
import math
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from hierarchy_from_taxonomy import (
    prepare_hierarchy_from_taxonomy,
    compute_distance_matrix_from_paths,
    build_distance_level_index,
)
from dataset import HierarchicalTextDataset, hierarchical_collate_fn
from model import HierarchicalMultiLabelModel
from hierarchical_infoNCE import NewInBatchHierarchyContrastiveLoss
from eval import evaluate_multilabel
from losses_multilabel import build_multilabel_loss

DEBUG_MAX_STEPS = 100
DEBUG_LOG_EVERY = 1000

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_label_to_id(labels_path: str) -> dict[str, int]:
    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            labels.append(name)
    unique_labels = sorted(set(labels))
    return {lab: idx for idx, lab in enumerate(unique_labels)}


def build_path_label_ids(paths, label_to_id):
    """
    Convert each path (node names / ids) into label-id sequence.
    Example:
        ['A','B','C'] -> [id(A), id(B), id(C)]
    """
    path_label_ids = []
    for p in paths:
        ids = []
        for node in p:
            if isinstance(node, int):
                nid = int(node)
            else:
                nid = label_to_id.get(node, None)
            if nid is None:
                raise ValueError(f"Cannot map node '{node}' to label id.")
            ids.append(nid)
        path_label_ids.append(ids)
    return path_label_ids


@torch.no_grad()
def build_path_prediction_score_matrix(
    logits: torch.Tensor,
    path_label_ids: list[list[int]],
) -> torch.Tensor:
    """
    logits: [B, num_labels]

    For each sample x and each path p:
        score(x, p) = average sigmoid probability over all labels in that path

    return:
        score_matrix: [B, num_paths]
    """
    probs = torch.sigmoid(logits)   # [B, num_labels]
    B = probs.size(0)
    P = len(path_label_ids)

    score_matrix = torch.empty(B, P, device=probs.device, dtype=probs.dtype)

    for pid, label_ids in enumerate(path_label_ids):
        idx = torch.tensor(label_ids, device=probs.device, dtype=torch.long)
        score_matrix[:, pid] = probs.index_select(1, idx).mean(dim=1)

    return score_matrix


def instance_contrastive_loss(text_proj, desc_proj, temperature: float = 0.07):
    """
    Symmetric InfoNCE between text_proj and desc_proj.
    """
    B = text_proj.size(0)
    t = F.normalize(text_proj, dim=-1)
    d = F.normalize(desc_proj, dim=-1)

    sim = (t @ d.t()) / temperature
    labels = torch.arange(B, device=text_proj.device)

    loss_t2d = F.cross_entropy(sim, labels)
    loss_d2t = F.cross_entropy(sim.t(), labels)

    return 0.5 * (loss_t2d + loss_d2t)


def multi_positive_contrastive_loss(
    text_proj,
    desc_proj,
    desc_owner,
    temperature: float = 0.07,
    pos_agg: str = "logsumexp",
):
    """
    text_proj: [B, D]
    desc_proj: [M, D]
    desc_owner: [M] mapping each description to its owning example in [0, B-1]
    """
    if desc_proj is None or desc_proj.numel() == 0 or desc_owner.numel() == 0:
        return torch.zeros((), device=text_proj.device, dtype=text_proj.dtype)

    t = F.normalize(text_proj, dim=-1)  # [B, D]
    d = F.normalize(desc_proj, dim=-1)  # [M, D]

    logits = (t @ d.t()) / temperature  # [B, M]
    denom = torch.logsumexp(logits, dim=1)  # [B]

    B = t.size(0)
    loss_terms = []
    for i in range(B):
        pos_mask = (desc_owner == i)
        if not pos_mask.any():
            continue

        if pos_agg == "avg":
            per_pos = -(logits[i, pos_mask] - denom[i])
            loss_terms.append(per_pos.mean())
        elif pos_agg == "logsumexp":
            num = torch.logsumexp(logits[i, pos_mask], dim=0)
            loss_terms.append(-(num - denom[i]))
        else:
            raise ValueError(f"Unknown pos_agg={pos_agg}. Use 'avg' or 'logsumexp'.")

    loss_t2d = (
        torch.stack(loss_terms).mean()
        if len(loss_terms) > 0
        else torch.zeros((), device=t.device, dtype=t.dtype)
    )

    logits_dt = logits.t()  # [M, B]
    denom_d = torch.logsumexp(logits_dt, dim=1)  # [M]
    idx = torch.arange(logits_dt.size(0), device=logits_dt.device)
    num_d = logits_dt[idx, desc_owner]
    loss_d2t = -(num_d - denom_d).mean()

    return 0.5 * (loss_t2d + loss_d2t)


# -----------------------------------------------------------------------------
# Optional legacy helper (not used in sample-specific debug)
# -----------------------------------------------------------------------------
@torch.no_grad()
def recall_at_k_candidates(
    text_embs: torch.Tensor,
    cand_path_embs: torch.Tensor,
    batch_paths,
    cand_ids: torch.Tensor,
    k: int = 5,
    tau: float = 0.07,
) -> float:
    t = F.normalize(text_embs, dim=-1)
    p = F.normalize(cand_path_embs, dim=-1)
    logits = (t @ p.t()) / tau
    topk = logits.topk(min(k, logits.size(1)), dim=-1).indices
    hits = 0
    total = len(batch_paths)
    cand_list = cand_ids.tolist()
    for i, gt in enumerate(batch_paths):
        if gt is None or len(gt) == 0:
            continue
        pred_local = set(topk[i].tolist())
        pred_global = {cand_list[j] for j in pred_local}
        if len(pred_global.intersection(set(gt))) > 0:
            hits += 1
    return hits / max(1, total)


# -----------------------------------------------------------------------------
# Negative sampler config + helpers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Negative sampler config + helpers
# -----------------------------------------------------------------------------
NEG_SAMPLER_CFG = {
    "WOS": {"neg_mul": 4, "neg_min": 8, "neg_max": 16},
    "BGC": {"neg_mul": 4, "neg_min": 12, "neg_max": 28},
    "NYT": {"neg_mul": 4, "neg_min": 16, "neg_max": 40},
}


def unique_in_order(ids):
    seen = set()
    out = []
    for x in ids:
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def get_negative_budget(
    dataset_name: str,
    num_gt: int,
    neg_mul: float | None = None,
    neg_min: int | None = None,
    neg_max: int | None = None,
) -> int:
    name = dataset_name.upper()
    if name not in NEG_SAMPLER_CFG:
        raise ValueError(
            f"Unknown dataset_name={dataset_name}. Expected one of {list(NEG_SAMPLER_CFG.keys())}."
        )

    cfg = NEG_SAMPLER_CFG[name]
    use_neg_mul = cfg["neg_mul"] if neg_mul is None else neg_mul
    use_neg_min = cfg["neg_min"] if neg_min is None else neg_min
    use_neg_max = cfg["neg_max"] if neg_max is None else neg_max
    budget = int(use_neg_mul * max(1, num_gt))
    budget = max(use_neg_min, min(use_neg_max, budget))
    return int(budget)


def round_robin_take_from_levels(
    level_to_items: dict[int, list[tuple[int, int, float]]],
    quota: int,
) -> list[tuple[int, int, float]]:
    """
    level_to_items:
        {distance_level: [(pid, dist, score), ...]}
    Each level list must already be sorted by score descending.

    Round-robin over levels from near to far:
        L1 first item, L2 first item, ..., then second round, etc.
    """
    if quota <= 0:
        return []

    levels = sorted(level_to_items.keys())
    ptr = {d: 0 for d in levels}
    chosen = []

    while len(chosen) < quota:
        progressed = False
        for d in levels:
            items = level_to_items[d]
            k = ptr[d]
            if k < len(items):
                chosen.append(items[k])
                ptr[d] += 1
                progressed = True
                if len(chosen) >= quota:
                    break
        if not progressed:
            break  # all levels exhausted

    return chosen


def _avg_stat(stat: dict, key_sum: str, key_count: str) -> float:
    return stat[key_sum] / max(1, stat[key_count])


def sample_candidates_for_one_example(
    gt_path_ids,
    distance_level_index,
    dataset_name: str,
    sample_scores: torch.Tensor,   # shape [num_paths]
    return_debug: bool = False,
    sample_neg_mul: float | None = None,
    sample_neg_min: int | None = None,
    sample_neg_max: int | None = None,
):
    """
    Clean version:
      1) distance -> level
      2) sort within each level by score desc
      3) round-robin across levels until quota is full

    sample_scores:
      path-level prediction score for ONE sample against ALL paths
      defined as the average sigmoid probability over labels in each path
    """
    gt_ids = unique_in_order(gt_path_ids)
    gt_set = set(gt_ids)

    if len(gt_ids) == 0:
        if return_debug:
            return [], {
                "num_gt": 0,
                "total_budget": 0,
                "provisional_quota": 0,
                "pre_dedup_selected": 0,
                "post_dedup_selected": 0,
                "final_negatives": 0,
                "final_avg_score": 0.0,
                "final_avg_dist": 0.0,
            }
        return []

    total_budget = get_negative_budget(
        dataset_name,
        len(gt_ids),
        neg_mul=sample_neg_mul,
        neg_min=sample_neg_min,
        neg_max=sample_neg_max,
    )
    provisional_quota = int(math.ceil(1.5 * total_budget / max(1, len(gt_ids))))

    # stats for selected provisional pool
    selected_stats = {}

    # stats for all reachable candidates (used when refill is needed)
    global_stats = {}

    pre_dedup_selected = 0

    def update_stats(stats_dict, pid: int, dist: int, score: float):
        if pid not in stats_dict:
            stats_dict[pid] = {
                "support": 0,
                "min_dist": dist,
                "dist_sum": 0.0,
                "dist_count": 0,
                "score_sum": 0.0,
                "score_count": 0,
            }

        stats_dict[pid]["support"] += 1
        stats_dict[pid]["min_dist"] = min(stats_dict[pid]["min_dist"], dist)
        stats_dict[pid]["dist_sum"] += float(dist)
        stats_dict[pid]["dist_count"] += 1
        stats_dict[pid]["score_sum"] += float(score)
        stats_dict[pid]["score_count"] += 1

    def avg_dist(stat):
        return _avg_stat(stat, "dist_sum", "dist_count")

    def avg_score(stat):
        return _avg_stat(stat, "score_sum", "score_count")

    # ---------------------------------------------------------
    # 1) Per-GT provisional sampling
    # ---------------------------------------------------------
    for g in gt_ids:
        level_map = distance_level_index[g]
        levels = [d for d in sorted(level_map.keys()) if d > 0]
        if len(levels) == 0:
            continue

        level_to_items = {}

        for d in levels:
            ids = [pid for pid in level_map.get(d, []) if pid not in gt_set]
            if len(ids) == 0:
                continue

            items = []
            for pid in ids:
                score = float(sample_scores[pid].item())
                items.append((pid, d, score))
                update_stats(global_stats, pid, d, score)

            # sort within same level by score descending
            items.sort(key=lambda x: (-x[2], x[0]))
            level_to_items[d] = items

        chosen = round_robin_take_from_levels(level_to_items, provisional_quota)
        pre_dedup_selected += len(chosen)

        for pid, dist, score in chosen:
            update_stats(selected_stats, pid, dist, score)

    provisional_neg_ids = [pid for pid in selected_stats.keys() if pid not in gt_set]

    # ---------------------------------------------------------
    # 2) Merge + dedup + truncate if too many
    # priority:
    #   higher support
    #   smaller min_dist
    #   larger avg_score
    #   smaller avg_dist
    # ---------------------------------------------------------
    provisional_neg_ids = sorted(
        provisional_neg_ids,
        key=lambda pid: (
            -selected_stats[pid]["support"],
            selected_stats[pid]["min_dist"],
            -avg_score(selected_stats[pid]),
            avg_dist(selected_stats[pid]),
            pid,
        ),
    )

    if len(provisional_neg_ids) > total_budget:
        provisional_neg_ids = provisional_neg_ids[:total_budget]

    # ---------------------------------------------------------
    # 3) Refill if not enough
    # ---------------------------------------------------------
    if len(provisional_neg_ids) < total_budget:
        selected_set = set(provisional_neg_ids)

        remaining_ids = [
            pid for pid in global_stats.keys()
            if pid not in gt_set and pid not in selected_set
        ]

        remaining_ids = sorted(
            remaining_ids,
            key=lambda pid: (
                -global_stats[pid]["support"],
                global_stats[pid]["min_dist"],
                -avg_score(global_stats[pid]),
                avg_dist(global_stats[pid]),
                pid,
            ),
        )

        need = total_budget - len(provisional_neg_ids)
        provisional_neg_ids.extend(remaining_ids[:need])

    candidate_ids = unique_in_order(list(gt_ids) + list(provisional_neg_ids))

    if not return_debug:
        return candidate_ids

    final_score_vals = []
    final_dist_vals = []
    for pid in provisional_neg_ids:
        src = selected_stats[pid] if pid in selected_stats else global_stats[pid]
        final_score_vals.append(avg_score(src))
        final_dist_vals.append(avg_dist(src))

    debug_info = {
        "num_gt": len(gt_ids),
        "total_budget": total_budget,
        "provisional_quota": provisional_quota,
        "pre_dedup_selected": pre_dedup_selected,
        "post_dedup_selected": len(selected_stats),
        "final_negatives": len(provisional_neg_ids),
        "final_avg_score": float(np.mean(final_score_vals)) if len(final_score_vals) > 0 else 0.0,
        "final_avg_dist": float(np.mean(final_dist_vals)) if len(final_dist_vals) > 0 else 0.0,
    }
    return candidate_ids, debug_info


# def get_staged_loss_weights(
#     dataset_name: str,
#     epoch_idx: int,
#     beta: float,
#     beta_desc: float,
# ) -> tuple[float, float, str]:
#     """
#     Stage multi-objective training by dataset depth:
#       WOS: epoch 1-3 BCE, 4-10 BCE+pathSimNCE, 11+ all losses
#       BGC: epoch 1-3 BCE, 4-15 BCE+pathSimNCE, 16+ all losses
#       NYT: epoch 1-5 BCE, 6-20 BCE+pathSimNCE, 21+ all losses
#     """
#     name = dataset_name.upper()
#     schedule = {
#         "WOS": {"bce_end": 3, "hcl_end": 10},
#         "BGC": {"bce_end": 3, "hcl_end": 15},
#         "NYT": {"bce_end": 5, "hcl_end": 20},
#     }
#
#     if name not in schedule:
#         return beta, beta_desc, "BCE + pathSimNCE + description"
#
#     bce_end = schedule[name]["bce_end"]
#     hcl_end = schedule[name]["hcl_end"]
#
#     if epoch_idx <= bce_end:
#         return 0.0, 0.0, "BCE only"
#     if epoch_idx <= hcl_end:
#         return beta, 0.0, "BCE + pathSimNCE"
#     return beta, beta_desc, "BCE + pathSimNCE + description"


# -----------------------------------------------------------------------------
# Sample-specific debug helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _sample_specific_debug_stats(
    text_hcl: torch.Tensor,
    path_hcl: torch.Tensor,
    batch_paths,
    candidate_ids_per_sample,
    hcl: NewInBatchHierarchyContrastiveLoss,
):
    """
    Compute sample-specific debug metrics averaged over the batch.
    """
    device = text_hcl.device
    t_all = F.normalize(text_hcl, dim=-1)
    p_all = F.normalize(path_hcl, dim=-1)
    tau = float(getattr(hcl, "tau", 0.07))

    entp_ratio_list = []
    entq_ratio_list = []
    kl_list = []
    r5_list = []
    cand_sizes = []

    sample0 = None

    for b, gt in enumerate(batch_paths):
        if gt is None or len(gt) == 0:
            continue

        cand_ids = unique_in_order(list(gt) + list(candidate_ids_per_sample[b]))
        if len(cand_ids) == 0:
            continue

        cand = torch.tensor(cand_ids, device=device, dtype=torch.long)
        cand_path_embs = p_all.index_select(0, cand)  # [C, D]

        logits = (t_all[b : b + 1] @ cand_path_embs.t()) / tau  # [1, C]
        log_p = F.log_softmax(logits, dim=-1).squeeze(0)  # [C]
        p = log_p.exp()

        q = hcl._build_q_for_candidates(
            gt_ids=gt,
            cand_ids=cand_ids,
            device=device,
            dtype=text_hcl.dtype,
        )

        C = int(len(cand_ids))
        logC = max(1e-12, math.log(max(2, C)))  # avoid degenerate divide
        ent_p = float((-(p * log_p).sum()).item())
        ent_q = float((-(q * torch.log(q + 1e-12)).sum()).item())
        kl = float((q * (torch.log(q + 1e-12) - log_p)).sum().item())

        topk = min(5, C)
        top_idx = torch.topk(logits.squeeze(0), k=topk).indices.tolist()
        pred_global = [cand_ids[j] for j in top_idx]
        hit = 1.0 if len(set(pred_global).intersection(set(gt))) > 0 else 0.0

        entp_ratio_list.append(ent_p / logC)
        entq_ratio_list.append(ent_q / logC)
        kl_list.append(kl)
        r5_list.append(hit)
        cand_sizes.append(C)

        if b == 0:
            top_pred_val, top_pred_idx = torch.topk(p, k=min(5, C))
            top_q_val, top_q_idx = torch.topk(q, k=min(5, C))
            sample0 = {
                "gt_ids": list(gt),
                "candidate_size": C,
                "top_pred_ids": [cand_ids[j] for j in top_pred_idx.tolist()],
                "top_pred_probs": [float(x) for x in top_pred_val.tolist()],
                "top_q_ids": [cand_ids[j] for j in top_q_idx.tolist()],
                "top_q_probs": [float(x) for x in top_q_val.tolist()],
                "r5_hit": bool(hit > 0.0),
            }

    if len(cand_sizes) == 0:
        return None

    return {
        "avg_candidate_size": float(np.mean(cand_sizes)),
        "min_candidate_size": int(np.min(cand_sizes)),
        "max_candidate_size": int(np.max(cand_sizes)),
        "entp_ratio": float(np.mean(entp_ratio_list)),
        "entq_ratio": float(np.mean(entq_ratio_list)),
        "kl": float(np.mean(kl_list)),
        "r5": float(np.mean(r5_list)),
        "sample0": sample0,
    }


###############################################################################
# Utility: Make Dataloader
###############################################################################
def make_dataloader(
    data_path: str,
    tokenizer,
    path_to_id,
    label_to_id,
    num_labels: int,
    batch_size: int,
    max_length: int,
    shuffle: bool,
):
    ds = HierarchicalTextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        label_to_id=label_to_id,
        num_labels=num_labels,
        text_key="text",
        path_key="path",  # assumes path = [[...], ...]
        label_key="label",
        desc_key="path_description",
        max_length=max_length,
    )
    collate = lambda batch: hierarchical_collate_fn(batch, num_labels=num_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _sanitize_run_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in tag)


def _metric_score(
    stats: dict[str, float],
    selection_metric: str,
    target_micro_f1: float,
    target_macro_f1: float,
) -> float:
    if selection_metric == "micro":
        return stats["micro_f1"]
    if selection_metric == "macro":
        return stats["macro_f1"]
    if selection_metric == "joint":
        return 0.5 * (stats["micro_f1"] + stats["macro_f1"])
    if selection_metric == "target":
        micro_ratio = stats["micro_f1"] / max(1e-8, target_micro_f1)
        macro_ratio = stats["macro_f1"] / max(1e-8, target_macro_f1)
        return min(micro_ratio, macro_ratio)
    raise ValueError(f"Unknown selection_metric={selection_metric}.")


def _build_bce_pos_weight_from_dataset(
    dataset: HierarchicalTextDataset,
    num_labels: int,
    power: float,
    max_weight: float,
) -> torch.Tensor | None:
    if power <= 0.0:
        return None

    counts = torch.zeros(num_labels, dtype=torch.float)
    for ex in dataset.data:
        label_ids = []
        if dataset.label_key in ex and ex[dataset.label_key]:
            labs = ex[dataset.label_key]
            if isinstance(labs, (list, tuple)):
                for lab in labs:
                    if isinstance(lab, str) and lab in dataset.label_to_id:
                        label_ids.append(dataset.label_to_id[lab])
        else:
            for p in dataset._get_paths_for_example(ex):
                for node in p:
                    if node in dataset.label_to_id:
                        label_ids.append(dataset.label_to_id[node])

        for label_id in set(label_ids):
            counts[int(label_id)] += 1.0

    num_examples = max(1, len(dataset.data))
    neg_counts = num_examples - counts
    pos_counts = counts.clamp_min(1.0)
    pos_weight = (neg_counts / pos_counts).clamp_min(1.0)
    pos_weight = pos_weight.pow(power).clamp(max=max_weight)
    return pos_weight


###############################################################################
# Train one epoch
###############################################################################
def train_one_epoch(
    model,
    hcl,
    loader,
    optimizer,
    scheduler,
    device,
    debug_model: bool = False,
    beta: float = 1,
    beta_desc: float = 0.1,
    desc_pos_agg: str = "logsumexp",
    max_steps: int | None = None,
    epoch_idx: int = 1,
    dataset_name: str | None = None,
    distance_level_index=None,
    path_label_ids=None,
    hcl_candidate_mode: str = "sample",
    hcl_space: str = "separate",
    multilabel_loss_name: str = "bce",
    bce_pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.25,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float | None = 0.05,
    asl_eps: float = 1e-8,
    asl_disable_grad: bool = False,
    sample_neg_mul: float | None = None,
    sample_neg_min: int | None = None,
    sample_neg_max: int | None = None,
):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_hcl = 0.0
    total_desc = 0.0
    steps = 0

    classification_loss_fn = build_multilabel_loss(
        multilabel_loss_name,
        bce_pos_weight=bce_pos_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        asl_gamma_neg=asl_gamma_neg,
        asl_gamma_pos=asl_gamma_pos,
        asl_clip=asl_clip,
        asl_eps=asl_eps,
        asl_disable_grad=asl_disable_grad,
    )

    # moving averages for debug
    ma_count = 0
    ma_avgC = 0.0
    ma_minC = 0.0
    ma_maxC = 0.0
    ma_entp_ratio = 0.0
    ma_entq_ratio = 0.0
    ma_kl = 0.0
    ma_r5 = 0.0
    ma_neg = 0.0
    ma_score = 0.0
    ma_dist = 0.0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_paths = batch["batch_paths"]
        desc_input_ids = batch["desc_input_ids"].to(device)
        desc_attention_mask = batch["desc_attention_mask"].to(device)
        desc_owner = batch["desc_owner"].to(device)
        use_hcl_loss = beta != 0.0
        use_desc_loss = beta_desc != 0.0

        # 1) forward first
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            desc_input_ids=desc_input_ids if use_desc_loss else None,
            desc_attention_mask=desc_attention_mask if use_desc_loss else None,
        )
        logits = outputs["logits"]
        text_proj = outputs["text_proj"]
        desc_proj = outputs["desc_proj"]
        text_hcl = outputs["text_hcl"]
        path_hcl = outputs["path_hcl"]
        if hcl_space == "raw":
            hcl_text_input = outputs["text_embs"]
            hcl_path_input = outputs["path_embs"]
        elif hcl_space == "separate":
            hcl_text_input = text_hcl
            hcl_path_input = path_hcl
        else:
            raise ValueError(f"Unknown hcl_space={hcl_space}. Use 'separate' or 'raw'.")

        # 2) build sample-specific candidates using CURRENT scores
        candidate_ids_per_sample = None
        sampler_debug_infos = None

        # if distance_level_index is not None and dataset_name is not None:
        #     with torch.no_grad():
        #         t_norm = F.normalize(text_hcl.detach(), dim=-1)  # [B, D]
        #         p_norm = F.normalize(path_hcl.detach(), dim=-1)  # [P, D]
        #         score_matrix = t_norm @ p_norm.t()  # [B, P]
        if use_hcl_loss and hcl_candidate_mode not in {"sample", "inbatch"}:
            raise ValueError(
                f"Unknown hcl_candidate_mode={hcl_candidate_mode}. Use 'sample' or 'inbatch'."
            )

        if use_hcl_loss and hcl_candidate_mode == "sample" and distance_level_index is not None and dataset_name is not None:
            if path_label_ids is None:
                raise ValueError("path_label_ids must be provided.")

            with torch.no_grad():
                score_matrix = build_path_prediction_score_matrix(
                    logits=logits.detach(),
                    path_label_ids=path_label_ids,
                )

            if debug_model:
                candidate_ids_per_sample = []
                sampler_debug_infos = []
                for b, gt_ids in enumerate(batch_paths):
                    cand_ids, dbg = sample_candidates_for_one_example(
                        gt_path_ids=gt_ids,
                        distance_level_index=distance_level_index,
                        dataset_name=dataset_name,
                        sample_scores=score_matrix[b],
                        return_debug=True,
                        sample_neg_mul=sample_neg_mul,
                        sample_neg_min=sample_neg_min,
                        sample_neg_max=sample_neg_max,
                    )
                    candidate_ids_per_sample.append(cand_ids)
                    sampler_debug_infos.append(dbg)
            else:
                candidate_ids_per_sample = [
                    sample_candidates_for_one_example(
                        gt_path_ids=gt_ids,
                        distance_level_index=distance_level_index,
                        dataset_name=dataset_name,
                        sample_scores=score_matrix[b],
                        return_debug=False,
                        sample_neg_mul=sample_neg_mul,
                        sample_neg_min=sample_neg_min,
                        sample_neg_max=sample_neg_max,
                    )
                    for b, gt_ids in enumerate(batch_paths)
                ]

        # 3) losses
        bce_loss = classification_loss_fn(logits, labels)
        if use_hcl_loss:
            h_loss = hcl(
                hcl_text_input,
                hcl_path_input,
                batch_paths,
                candidate_ids_per_sample=candidate_ids_per_sample,
            )
        else:
            h_loss = torch.zeros((), device=device, dtype=bce_loss.dtype)

        if use_desc_loss:
            desc_loss = multi_positive_contrastive_loss(
                text_proj,
                desc_proj,
                desc_owner,
                temperature=0.07,
                pos_agg=desc_pos_agg,
            )
        else:
            desc_loss = torch.zeros((), device=device, dtype=bce_loss.dtype)

        # ---------------------------------------------------------
        # Sample-specific debug block
        # ---------------------------------------------------------
        if debug_model and candidate_ids_per_sample is not None:
            dbg = _sample_specific_debug_stats(
                text_hcl=hcl_text_input,
                path_hcl=hcl_path_input,
                batch_paths=batch_paths,
                candidate_ids_per_sample=candidate_ids_per_sample,
                hcl=hcl,
            )

            if dbg is not None:
                ma_count += 1
                ma_avgC += dbg["avg_candidate_size"]
                ma_minC += dbg["min_candidate_size"]
                ma_maxC += dbg["max_candidate_size"]
                ma_entp_ratio += dbg["entp_ratio"]
                ma_entq_ratio += dbg["entq_ratio"]
                ma_kl += dbg["kl"]
                ma_r5 += dbg["r5"]

            if sampler_debug_infos is not None and len(sampler_debug_infos) > 0:
                neg_counts = [x["final_negatives"] for x in sampler_debug_infos]
                avg_scores = [x["final_avg_score"] for x in sampler_debug_infos]
                avg_dists = [x["final_avg_dist"] for x in sampler_debug_infos]

                ma_neg += float(np.mean(neg_counts))
                ma_score += float(np.mean(avg_scores))
                ma_dist += float(np.mean(avg_dists))

            should_print = ((steps + 1) % DEBUG_LOG_EVERY == 0) or (steps == 0)
            if should_print and ma_count > 0:
                denom = max(1, ma_count)
                print(
                    f"[step {steps + 1}] "
                    f"avgC={ma_avgC / denom:.2f} "
                    f"minC={ma_minC / denom:.2f} "
                    f"maxC={ma_maxC / denom:.2f} "
                    f"avgNeg={ma_neg / denom:.2f} "
                    f"avgScore={ma_score / denom:.4f} "
                    f"avgDist={ma_dist / denom:.3f} "
                    f"ent_p/logC={ma_entp_ratio / denom:.3f} "
                    f"ent_q/logC={ma_entq_ratio / denom:.3f} "
                    f"KL={ma_kl / denom:.3f} "
                    f"R@5={ma_r5 / denom:.3f}"
                )

                if dbg is not None and dbg["sample0"] is not None:
                    s0 = dbg["sample0"]
                    print("----- SAMPLE 0 DEBUG -----")
                    print(f"gt_ids: {s0['gt_ids']}")
                    print(f"candidate_size: {s0['candidate_size']}")
                    print(f"top_pred_ids: {s0['top_pred_ids']}")
                    print(f"top_pred_probs: {[round(x, 4) for x in s0['top_pred_probs']]}")
                    print(f"top_q_ids: {s0['top_q_ids']}")
                    print(f"top_q_probs: {[round(x, 4) for x in s0['top_q_probs']]}")
                    print(f"R@5 hit: {s0['r5_hit']}")

                if sampler_debug_infos is not None and len(sampler_debug_infos) > 0:
                    sd0 = sampler_debug_infos[0]
                    print("----- SAMPLER 0 DEBUG -----")
                    print(
                        f"num_gt={sd0['num_gt']} "
                        f"budget={sd0['total_budget']} "
                        f"prov_quota={sd0['provisional_quota']} "
                        f"pre_dedup={sd0['pre_dedup_selected']} "
                        f"post_dedup={sd0['post_dedup_selected']} "
                        f"final_neg={sd0['final_negatives']} "
                        f"final_avg_score={sd0['final_avg_score']:.4f} "
                        f"final_avg_dist={sd0['final_avg_dist']:.3f}"
                    )

                # reset moving averages
                ma_count = 0
                ma_avgC = 0.0
                ma_minC = 0.0
                ma_maxC = 0.0
                ma_entp_ratio = 0.0
                ma_entq_ratio = 0.0
                ma_kl = 0.0
                ma_r5 = 0.0
                ma_neg = 0.0
                ma_score = 0.0
                ma_dist = 0.0

        loss = bce_loss + beta * h_loss + beta_desc * desc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_hcl += h_loss.item()
        total_desc += desc_loss.item()

        steps += 1
        if max_steps is not None and steps >= max_steps:
            print(f"Reached max steps ({max_steps}). Stopping epoch early.")
            break

    return {
        "loss": total_loss / steps,
        "bce": total_bce / steps,
        "hcl": total_hcl / steps,
        "desc": total_desc / steps,
    }


###############################################################################
# Full train/eval for a dataset
###############################################################################
def run_dataset(
    dataset_name: str,
    train_path: str,
    val_path: str,
    test_path: str,
    taxonomy_path: str,
    labels_path: str,
    encoder_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 256,
    epochs: int = 10,
    patience: int = 3,
    lr: float = 2e-5,
    head_lr: float = 1e-4,
    beta: float = 1.0,
    beta_desc: float = 0.1,
    desc_pos_agg: str = "logsumexp",
    tau: float = 0.07,
    alpha: float = 2.0,
    gamma: float = 0.3,
    lam: float = 0.5,
    threshold: float = 0.5,
    hierarchy_decode: bool = False,
    debug_model: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold_sweep: bool = False,
    threshold_sweep_min: float = 0.05,
    threshold_sweep_max: float = 0.95,
    threshold_sweep_steps: int = 9,
    test_use_best_threshold: bool = False,
    hcl_candidate_mode: str = "sample",
    hcl_space: str = "separate",
    selection_metric: str = "micro",
    target_micro_f1: float = 0.8216,
    target_macro_f1: float = 0.7188,
    multilabel_loss_name: str = "bce",
    bce_pos_weight_power: float = 0.0,
    bce_pos_weight_max: float = 5.0,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.25,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float | None = 0.05,
    asl_eps: float = 1e-8,
    asl_disable_grad: bool = False,
    sample_neg_mul: float | None = None,
    sample_neg_min: int | None = None,
    sample_neg_max: int | None = None,
    run_tag: str | None = None,
    eval_only_checkpoint: str | None = None,
    eval_only_threshold_path: str | None = None,
):
    print(f"\n================= Running {dataset_name} =================\n")

    if debug_model:
        print("Debug mode enabled. Using small batch size and epochs.")
        batch_size = 32
        epochs = 2
        patience = 1

    # ---------------------------------------------------------
    # Load taxonomy + build hierarchy
    # ---------------------------------------------------------
    paths, path_to_id, w_matrix = prepare_hierarchy_from_taxonomy(
        taxonomy_path=taxonomy_path,
        labels_path=labels_path,
        root_name="Root",
        lam=lam,
    )

    # Precompute path-distance index for sample-specific candidate sampling
    dist_matrix = compute_distance_matrix_from_paths(paths)
    distance_level_index = build_distance_level_index(dist_matrix)

    # label_to_id = load_label_to_id(labels_path)
    # num_labels = len(label_to_id)
    # num_paths = len(paths)
    # print(f"{dataset_name}: Num label nodes = {num_labels}, Num paths = {len(paths)}")
    label_to_id = load_label_to_id(labels_path)
    num_labels = len(label_to_id)
    num_paths = len(paths)
    path_label_ids = build_path_label_ids(paths, label_to_id)
    print(f"{dataset_name}: Num label nodes = {num_labels}, Num paths = {len(paths)}")

    def _to_label_id(x):
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            return label_to_id.get(x, None)
        return None

    ancestors = [set() for _ in range(num_labels)]
    for p in paths:
        p_ids = []
        for node in p:
            nid = _to_label_id(node)
            if nid is None:
                p_ids = []
                break
            p_ids.append(int(nid))
        if not p_ids:
            continue

        for idx, lab_id in enumerate(p_ids):
            if idx == 0:
                continue
            for anc_id in p_ids[:idx]:
                ancestors[lab_id].add(anc_id)

    ancestors = [sorted(list(s)) for s in ancestors]

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ---------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------
    train_loader = make_dataloader(
        data_path=train_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        label_to_id=label_to_id,
        num_labels=num_labels,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )
    val_loader = make_dataloader(
        data_path=val_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        label_to_id=label_to_id,
        num_labels=num_labels,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    test_loader = make_dataloader(
        data_path=test_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        label_to_id=label_to_id,
        num_labels=num_labels,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    bce_pos_weight = None
    if multilabel_loss_name.lower() == "bce":
        bce_pos_weight = _build_bce_pos_weight_from_dataset(
            dataset=train_loader.dataset,
            num_labels=num_labels,
            power=bce_pos_weight_power,
            max_weight=bce_pos_weight_max,
        )
        if bce_pos_weight is not None:
            print(
                "Using BCE pos_weight: "
                f"power={bce_pos_weight_power:.3f}, max={bce_pos_weight_max:.3f}, "
                f"mean={bce_pos_weight.mean().item():.3f}, max_actual={bce_pos_weight.max().item():.3f}"
            )
            bce_pos_weight = bce_pos_weight.to(device)

    print(f"Classification loss: {multilabel_loss_name}")
    if multilabel_loss_name.lower() == "focal":
        print(f"Focal config: gamma={focal_gamma:.3f}, alpha={focal_alpha}")
    elif multilabel_loss_name.lower() == "asl":
        print(
            "ASL config: "
            f"gamma_neg={asl_gamma_neg:.3f}, gamma_pos={asl_gamma_pos:.3f}, "
            f"clip={asl_clip}, eps={asl_eps:.1e}, disable_grad={asl_disable_grad}"
        )

    # ---------------------------------------------------------
    # Model + losses
    # ---------------------------------------------------------
    model = HierarchicalMultiLabelModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
        num_paths=num_paths,
        dropout=0.1,
        use_cls_token=True,
    ).to(device)

    if eval_only_checkpoint:
        print(f"\nEval-only mode: loading checkpoint from {eval_only_checkpoint}")
        ckpt = torch.load(eval_only_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt_encoder = ckpt.get("encoder_name", None)
            if ckpt_encoder is not None and ckpt_encoder != encoder_name:
                raise RuntimeError(
                    f"Checkpoint encoder_name '{ckpt_encoder}' does not match current encoder_name '{encoder_name}'. "
                    f"Please rerun with --encoder_name {ckpt_encoder} or use the correct checkpoint."
                )
            state = ckpt["state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state)

        th_test = float(threshold)
        if eval_only_threshold_path:
            with open(eval_only_threshold_path, "r", encoding="utf-8") as f:
                th_test = float(f.read().strip())
            print(f"Using eval-only threshold: {th_test:.3f}")

        test_stats = evaluate_multilabel(
            model=model,
            loader=test_loader,
            device=device,
            threshold=th_test,
            ancestors=ancestors,
            hierarchy_decode=hierarchy_decode,
        )

        print(
            f"\n==== Test results for {dataset_name} ====\n"
            f"micro_P={test_stats['micro_precision']:.4f}\n"
            f"micro_R={test_stats['micro_recall']:.4f}\n"
            f"micro_F1={test_stats['micro_f1']:.4f}\n"
            f"macro_F1={test_stats['macro_f1']:.4f}\n"
        )

        return {
            "dataset": dataset_name,
            "test_micro_f1": test_stats["micro_f1"],
            "test_macro_f1": test_stats["macro_f1"],
        }

    hcl = NewInBatchHierarchyContrastiveLoss(
        w_matrix=w_matrix.to(device),
        tau=tau,
        alpha=alpha,
        gamma=gamma,
        reduction="mean",
    ).to(device)

    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": model.encoder.parameters(), "lr": lr, "weight_decay": 0.01},
    #         {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": 0.01},
    #         {"params": model.projection_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
    #         {"params": model.hcl_text_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
    #         {"params": model.hcl_path_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
    #         {"params": model.path_embedding.parameters(), "lr": head_lr, "weight_decay": 0.00},
    #     ]
    # )
    optimizer = torch.optim.AdamW(
        [
            {"name": "encoder", "params": model.encoder.parameters(), "lr": lr, "weight_decay": 0.01},
            {"name": "classifier", "params": model.classifier.parameters(), "lr": head_lr, "weight_decay": 0.01},
            {"name": "projection_head", "params": model.projection_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
            {"name": "hcl_text_head", "params": model.hcl_text_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
            {"name": "hcl_path_head", "params": model.hcl_path_head.parameters(), "lr": head_lr, "weight_decay": 0.01},
            {"name": "path_embedding", "params": model.path_embedding.parameters(), "lr": head_lr, "weight_decay": 0.00},
        ]
    )

    # ---------------------------------------------------------
    # Scheduler: warmup + linear decay
    # ---------------------------------------------------------
    # total_training_steps = (
    #     DEBUG_MAX_STEPS if debug_model else epochs * len(train_loader)
    # )
    # warmup_steps = int(0.1 * total_training_steps)
    #
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_training_steps,
    # )
    # ---------------------------------------------------------
    # Scheduler: ReduceLROnPlateau (monitor validation micro-F1)
    # ---------------------------------------------------------
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="max",  # micro-F1 越大越好
    #     factor=0.5,  # 学习率降到原来的一�?
    #     patience=2,  # val micro-F1 连续 2 �?epoch 没明显提升就�?
    #     threshold=1e-3,  # 提升幅度太小就不算真正提�?
    #     threshold_mode="rel",
    #     min_lr=1e-6,
    # )
    # total_training_steps = epochs * len(train_loader)
    # warmup_steps = int(0.1 * total_training_steps)
    #
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_training_steps,
    # )
    steps_per_epoch = min(len(train_loader), DEBUG_MAX_STEPS) if debug_model else len(train_loader)
    total_training_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(0.1 * total_training_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---------------------------------------------------------
    # Early stopping
    # ---------------------------------------------------------
    best_selection_score = -float("inf")
    epochs_no_improve = 0

    os.makedirs("checkpoints", exist_ok=True)
    enc_tag = encoder_name.replace("/", "_")
    if run_tag is None:
        run_tag = os.environ.get("SLURM_JOB_ID", "").strip()
    tag_suffix = f"_{_sanitize_run_tag(run_tag)}" if run_tag else ""
    save_path = f"checkpoints/{dataset_name}_{enc_tag}{tag_suffix}_best_model.pt"
    best_th_path = f"checkpoints/{dataset_name}_{enc_tag}{tag_suffix}_best_threshold.txt"
    print(f"Checkpoint path: {save_path}")
    print(f"Selection metric: {selection_metric}")

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        # epoch_beta, epoch_beta_desc, loss_stage = get_staged_loss_weights(
        #     dataset_name=dataset_name,
        #     epoch_idx=epoch,
        #     beta=beta,
        #     beta_desc=beta_desc,
        # )
        # epoch_beta = beta
        # epoch_beta_desc = beta_desc
        # loss_stage = "Fixed loss weights"
        # print(
        #     f"Loss stage: {loss_stage} "
        #     f"(beta={epoch_beta:.4f}, beta_desc={epoch_beta_desc:.4f})"
        # )
        print(f"Fixed loss weights: beta={beta:.4f}, beta_desc={beta_desc:.4f}")

        train_stats = train_one_epoch(
            model=model,
            hcl=hcl,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            beta=beta,
            beta_desc=beta_desc,
            desc_pos_agg=desc_pos_agg,
            debug_model=debug_model,
            max_steps=DEBUG_MAX_STEPS if debug_model else None,
            epoch_idx=epoch,
            dataset_name=dataset_name,
            distance_level_index=distance_level_index,
            path_label_ids=path_label_ids,
            hcl_candidate_mode=hcl_candidate_mode,
            hcl_space=hcl_space,
            multilabel_loss_name=multilabel_loss_name,
            bce_pos_weight=bce_pos_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
            asl_eps=asl_eps,
            asl_disable_grad=asl_disable_grad,
            sample_neg_mul=sample_neg_mul,
            sample_neg_min=sample_neg_min,
            sample_neg_max=sample_neg_max,
        )

        print(
            f"Train Loss={train_stats['loss']:.4f} "
            f"(BCE={train_stats['bce']:.4f}, HCL={train_stats['hcl']:.4f}, DESC={train_stats['desc']:.4f})"
        )

        # ---- Validation ----
        if threshold_sweep:
            ths = np.linspace(threshold_sweep_min, threshold_sweep_max, threshold_sweep_steps)
            best = None
            best_th = float(threshold)
            for th in ths:
                stats_th = evaluate_multilabel(
                    model=model,
                    loader=val_loader,
                    device=device,
                    threshold=float(th),
                    ancestors=ancestors,
                    hierarchy_decode=hierarchy_decode,
                )
                stats_th_score = _metric_score(
                    stats_th,
                    selection_metric=selection_metric,
                    target_micro_f1=target_micro_f1,
                    target_macro_f1=target_macro_f1,
                )
                best_score = (
                    -float("inf")
                    if best is None
                    else _metric_score(
                        best,
                        selection_metric=selection_metric,
                        target_micro_f1=target_micro_f1,
                        target_macro_f1=target_macro_f1,
                    )
                )
                if best is None or stats_th_score > best_score:
                    best = stats_th
                    best_th = float(th)
            val_stats = best
            print(f"Val (sweep): best_threshold={best_th:.3f}")
        else:
            val_stats = evaluate_multilabel(
                model=model,
                loader=val_loader,
                device=device,
                threshold=threshold,
                ancestors=ancestors,
                hierarchy_decode=hierarchy_decode,
            )
            best_th = float(threshold)

        print(
            f"Val: micro_P={val_stats['micro_precision']:.4f} "
            f"micro_R={val_stats['micro_recall']:.4f} "
            f"micro_F1={val_stats['micro_f1']:.4f} "
            f"macro_F1={val_stats['macro_f1']:.4f}"
        )

        # if scheduler is not None:
        #     scheduler.step(val_stats["micro_f1"])
        #
        current_lrs = " | ".join(
            f"{group.get('name', f'group{i}')}={group['lr']:.8e}"
            for i, group in enumerate(optimizer.param_groups)
        )
        print(f"Current LRs: {current_lrs}")

        selection_score = _metric_score(
            val_stats,
            selection_metric=selection_metric,
            target_micro_f1=target_micro_f1,
            target_macro_f1=target_macro_f1,
        )

        if selection_score > best_selection_score + 1e-6:
            best_selection_score = selection_score
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder_name": encoder_name,
                },
                save_path,
            )
            with open(best_th_path, "w", encoding="utf-8") as f:
                f.write(f"{best_th}\n")
            print(f"New best selection score = {best_selection_score:.4f} (saved model)")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if not debug_model and epochs_no_improve >= patience:
            print(f"Early stopping triggered (patience={patience})")
            break

    # ---------------------------------------------------------
    # Load best model + Test
    # ---------------------------------------------------------
    print("\nLoading best model for testing...")
    ckpt = torch.load(save_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt_encoder = ckpt.get("encoder_name", None)
        if ckpt_encoder is not None and ckpt_encoder != encoder_name:
            raise RuntimeError(
                f"Checkpoint encoder_name '{ckpt_encoder}' does not match current encoder_name '{encoder_name}'. "
                f"Please rerun with --encoder_name {ckpt_encoder} or use the correct checkpoint."
            )
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)

    th_test = float(threshold)
    if test_use_best_threshold:
        if not threshold_sweep:
            print(
                "test_use_best_threshold is set but threshold_sweep was disabled; falling back to --threshold."
            )
        else:
            try:
                with open(best_th_path, "r", encoding="utf-8") as f:
                    th_test = float(f.read().strip())
                print(f"Using best validation threshold for test: {th_test:.3f}")
            except Exception as e:
                print(
                    f"Could not read best threshold from {best_th_path}: {e}. Falling back to --threshold."
                )

    test_stats = evaluate_multilabel(
        model=model,
        loader=test_loader,
        device=device,
        threshold=th_test,
        ancestors=ancestors,
        hierarchy_decode=hierarchy_decode,
    )

    print(
        f"\n==== Test results for {dataset_name} ====\n"
        f"micro_P={test_stats['micro_precision']:.4f}\n"
        f"micro_R={test_stats['micro_recall']:.4f}\n"
        f"micro_F1={test_stats['micro_f1']:.4f}\n"
        f"macro_F1={test_stats['macro_f1']:.4f}\n"
    )

    return {
        "dataset": dataset_name,
        "test_micro_f1": test_stats["micro_f1"],
        "test_macro_f1": test_stats["macro_f1"],
    }


###############################################################################
# Main runner
###############################################################################
def resolve_device(device_str: str) -> torch.device:
    """
    Resolve device from a string. Supports: 'auto', 'cpu', 'cuda', 'mps'.
    'auto' prefers cuda, then mps, then cpu.
    """
    ds = (device_str or "auto").lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    global DEBUG_MAX_STEPS, DEBUG_LOG_EVERY

    parser = argparse.ArgumentParser(
        description="Hierarchical Multi-Label Text Classification Training"
    )
    parser.add_argument(
        "--hierarchy_decode",
        action="store_true",
        help="Apply hierarchy-consistent decoding (ancestor closure) after thresholding during eval.",
    )
    parser.add_argument(
        "--desc_pos_agg",
        type=str,
        default="logsumexp",
        choices=["logsumexp", "avg"],
        help="How to aggregate multiple positive descriptions per example in DESC loss.",
    )

    # Paths / dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BGC",
        help="Dataset name used for logging/checkpoint naming.",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/BGC/bgc_train.json",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/BGC/bgc_val.json",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/BGC/bgc_test.json",
    )
    parser.add_argument(
        "--taxonomy_path",
        type=str,
        default="data/BGC/bgc.taxonomy",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="data/BGC/labels.txt",
    )

    # Model / tokenizer
    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use_cls_token",
        action="store_true",
        help="Use [CLS] token embedding. If not set, model may use pooling (per model.py).",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5, help="Encoder learning rate.")
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-4,
        help="LR for classifier/projection/path_embedding.",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for HCL.")
    parser.add_argument("--beta_desc", type=float, default=0.1, help="Weight for DESC.")
    parser.add_argument("--tau", type=float, default=0.07, help="Temperature for HCL.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Target sharpening power for HCL.",
    )
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Hierarchy similarity decay lambda for w_matrix.",
    )
    parser.add_argument(
        "--hcl_candidate_mode",
        type=str,
        default="sample",
        choices=["sample", "inbatch"],
        help="Use sample-specific HCL candidates or old in-batch GT-union candidates.",
    )
    parser.add_argument(
        "--hcl_space",
        type=str,
        default="separate",
        choices=["separate", "raw"],
        help="Use separate HCL heads or raw model embeddings for HCL.",
    )
    parser.add_argument(
        "--sample_neg_mul",
        type=float,
        default=None,
        help="Override sample-mode negative budget multiplier for this run.",
    )
    parser.add_argument(
        "--sample_neg_min",
        type=int,
        default=None,
        help="Override sample-mode minimum negative budget for this run.",
    )
    parser.add_argument(
        "--sample_neg_max",
        type=int,
        default=None,
        help="Override sample-mode maximum negative budget for this run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for multilabel evaluation.",
    )

    # Debug / logging
    parser.add_argument(
        "--debug_model",
        action="store_true",
        help="Enable debug mode (fixed max steps, extra logging).",
    )
    parser.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="If set, sweep thresholds on val to find best micro-F1 each epoch",
    )
    parser.add_argument("--threshold_sweep_min", type=float, default=0.05)
    parser.add_argument("--threshold_sweep_max", type=float, default=0.95)
    parser.add_argument("--threshold_sweep_steps", type=int, default=9)
    parser.add_argument(
        "--test_use_best_threshold",
        action="store_true",
        help="If set, evaluate test using the best threshold found on validation (requires --threshold_sweep).",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="micro",
        choices=["micro", "macro", "joint", "target"],
        help="Metric used for threshold selection, checkpointing, and early stopping.",
    )
    parser.add_argument("--target_micro_f1", type=float, default=0.8216)
    parser.add_argument("--target_macro_f1", type=float, default=0.7188)
    parser.add_argument(
        "--multilabel_loss",
        type=str,
        default="bce",
        choices=["bce", "focal", "asl"],
        help="Classification loss for multi-label prediction head.",
    )
    parser.add_argument(
        "--bce_pos_weight_power",
        type=float,
        default=0.0,
        help="If >0, applies softened inverse-frequency BCE positive weights.",
    )
    parser.add_argument(
        "--bce_pos_weight_max",
        type=float,
        default=5.0,
        help="Maximum BCE positive weight after applying bce_pos_weight_power.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for focal loss. Set negative to disable alpha balancing.",
    )
    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help="Negative focusing gamma for ASL.",
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=1.0,
        help="Positive focusing gamma for ASL.",
    )
    parser.add_argument(
        "--asl_clip",
        type=float,
        default=0.05,
        help="Asymmetric clipping value for ASL. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--asl_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for ASL.",
    )
    parser.add_argument(
        "--asl_disable_grad",
        action="store_true",
        help="Disable torch grad during the ASL focal-weight computation.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional checkpoint filename tag. Defaults to SLURM_JOB_ID when available.",
    )
    parser.add_argument(
        "--eval_only_checkpoint",
        type=str,
        default=None,
        help="If set, skip training and evaluate this checkpoint on the test split.",
    )
    parser.add_argument(
        "--eval_only_threshold_path",
        type=str,
        default=None,
        help="Optional threshold file to use with --eval_only_checkpoint.",
    )
    parser.add_argument("--debug_max_steps", type=int, default=DEBUG_MAX_STEPS)
    parser.add_argument("--debug_log_every", type=int, default=DEBUG_LOG_EVERY)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    args = parser.parse_args()

    set_seed(args.seed)

    DEBUG_MAX_STEPS = int(args.debug_max_steps)
    DEBUG_LOG_EVERY = int(args.debug_log_every)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    stats = run_dataset(
        dataset_name=args.dataset_name,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        taxonomy_path=args.taxonomy_path,
        labels_path=args.labels_path,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        head_lr=args.head_lr,
        beta=args.beta,
        beta_desc=args.beta_desc,
        desc_pos_agg=args.desc_pos_agg,
        tau=args.tau,
        alpha=args.alpha,
        gamma=args.gamma,
        lam=args.lam,
        threshold=args.threshold,
        threshold_sweep=args.threshold_sweep,
        threshold_sweep_min=args.threshold_sweep_min,
        threshold_sweep_max=args.threshold_sweep_max,
        threshold_sweep_steps=args.threshold_sweep_steps,
        debug_model=args.debug_model,
        device=device,
        test_use_best_threshold=args.test_use_best_threshold,
        hierarchy_decode=args.hierarchy_decode,
        hcl_candidate_mode=args.hcl_candidate_mode,
        hcl_space=args.hcl_space,
        selection_metric=args.selection_metric,
        target_micro_f1=args.target_micro_f1,
        target_macro_f1=args.target_macro_f1,
        multilabel_loss_name=args.multilabel_loss,
        bce_pos_weight_power=args.bce_pos_weight_power,
        bce_pos_weight_max=args.bce_pos_weight_max,
        focal_gamma=args.focal_gamma,
        focal_alpha=None if args.focal_alpha < 0 else args.focal_alpha,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=None if args.asl_clip <= 0 else args.asl_clip,
        asl_eps=args.asl_eps,
        asl_disable_grad=args.asl_disable_grad,
        sample_neg_mul=args.sample_neg_mul,
        sample_neg_min=args.sample_neg_min,
        sample_neg_max=args.sample_neg_max,
        run_tag=args.run_tag,
        eval_only_checkpoint=args.eval_only_checkpoint,
        eval_only_threshold_path=args.eval_only_threshold_path,
    )

    print("\n============== Summary ==============\n")
    print(
        f"{stats['dataset']}: micro-F1={stats['test_micro_f1']:.4f}, macro-F1={stats['test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

