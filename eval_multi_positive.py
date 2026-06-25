from __future__ import annotations
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn


@torch.no_grad()
def evaluate_multilabel(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    debug_model: bool = False,
    ancestors=None,
    hierarchy_decode: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a multi-label model on precision, recall, micro-F1, macro-F1.

    Assumes each batch from `loader` is a dict with:
        - "input_ids":   [B, L]
        - "attention_mask": [B, L]
        - "labels":      [B, P]  multi-hot ground truth (0/1)
    The model is assumed to return a dict with:
        - "logits": [B, P] (before sigmoid)

    Args:
        model:      your HierarchicalMultiLabelModel.
        loader:     DataLoader over the evaluation split.
        device:     torch.device("cuda") or torch.device("cpu").
        threshold:  decision threshold on sigmoid probabilities.

    Returns:
        dict with:
            "micro_precision"
            "micro_recall"
            "micro_f1"
            "macro_f1"
    """
    model.eval()

    if debug_model:
        print("Debug mode enabled. Using small batch size and epochs.")
        batch_size = 1
        epochs = 1

    all_labels = []
    all_preds = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # [B, P]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # [B, P]

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # ---- Hierarchy-consistent decoding (ancestor closure) ----
        # If a child label is predicted positive, force all its ancestors to be positive.
        if hierarchy_decode and ancestors is not None:
            import numpy as np
            if isinstance(preds, np.ndarray):
                pred_arr = preds
            else:
                pred_arr = preds.detach().cpu().numpy()

            N, L = pred_arr.shape
            for i in range(N):
                pos = np.where(pred_arr[i] > 0)[0]
                for lab in pos:
                    for a in ancestors[lab]:
                        pred_arr[i, a] = 1

            if isinstance(preds, np.ndarray):
                preds = pred_arr
            else:
                preds = preds.new_tensor(pred_arr)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)  # [N, P]
    y_pred = np.concatenate(all_preds, axis=0)   # [N, P]

    # --- Micro metrics ---
    # Flatten all label-instance pairs
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # --- Macro F1 ---
    # Compute per-label F1, then average
    num_labels = y_true.shape[1]
    f1_per_label = []

    for j in range(num_labels):
        y_t = y_true[:, j]
        y_p = y_pred[:, j]

        tp_j = np.sum((y_t == 1) & (y_p == 1))
        fp_j = np.sum((y_t == 0) & (y_p == 1))
        fn_j = np.sum((y_t == 1) & (y_p == 0))

        if tp_j == 0 and fp_j == 0 and fn_j == 0:
            # Label never appears in truth or pred -> skip or count as 0
            # Here we skip to avoid dividing by zero.
            continue

        prec_j = tp_j / (tp_j + fp_j + 1e-8)
        rec_j = tp_j / (tp_j + fn_j + 1e-8)
        if prec_j + rec_j == 0:
            f1_j = 0.0
        else:
            f1_j = 2 * prec_j * rec_j / (prec_j + rec_j + 1e-8)

        f1_per_label.append(f1_j)

    if len(f1_per_label) == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = float(np.mean(f1_per_label))

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
    }