from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import json
import random

import torch
from torch.utils.data import Dataset


class HierarchicalTextDataset(Dataset):
    """
    Generic dataset for WoS / BGC / NYT style hierarchical label paths.

    Expects each line in data_path to be a JSON object with:
        - "token": text string
        - "path":  [ [level1, level2, ...], ... ]   (preferred)
          OR
        - "label": [level1, level2, ...]           (fallback single-path)
        - "path_description": list of description strings aligned with "path" (optional),
          used to build an instance-specific path description.

    Args:
        data_path:   path to the JSONL file (train / val / test).
        tokenizer:   a HuggingFace-style tokenizer.
        path_to_id:  dict mapping full path tuples -> global path_id (for hierarchy-aware loss).
        label_to_id: dict mapping label node names (strings) -> label_id (for BCE).
        num_labels:  total number of label nodes (for BCE).
        text_key:    key for text field ("token").
        path_key:    key for path field ("path").
        label_key:   fallback key ("label").
        desc_key:    key for path description field ("path_description").
        max_length:  max sequence length for tokenization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        path_to_id: Dict[Tuple[str, ...], int],
        label_to_id: Dict[str, int],
        num_labels: int,
        text_key: str = "token",
        path_key: str = "path",
        label_key: str = "label",
        desc_key: str = "path_description",
        max_length: int = 512,
    ):
        self.data: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.path_to_id = path_to_id
        self.label_to_id = label_to_id
        self.num_labels = num_labels  # number of label nodes (for BCE), not paths
        self.text_key = text_key
        self.path_key = path_key
        self.label_key = label_key
        self.desc_key = desc_key
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if self.text_key not in obj:
                    continue
                self.data.append(obj)

    def __len__(self) -> int:
        return len(self.data)

    def _get_paths_for_example(self, ex: Dict[str, Any]) -> List[List[str]]:
        """
        Returns a list of label paths (each is a list[str]) for this example.
        """
        # Preferred: "path" is a list of paths
        if self.path_key in ex and ex[self.path_key]:
            paths_raw = ex[self.path_key]
            if isinstance(paths_raw, list) and len(paths_raw) > 0:
                return [
                    p for p in paths_raw
                    if isinstance(p, (list, tuple)) and len(p) > 0
                ]

        # Fallback: "label" as a single path
        if self.label_key in ex and ex[self.label_key]:
            labels = ex[self.label_key]
            if isinstance(labels, (list, tuple)) and len(labels) > 0:
                return [list(labels)]

        return []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        text = ex[self.text_key]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)        # [L]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [L]

        # Gather global path IDs for this example
        paths_raw = self._get_paths_for_example(ex)
        path_ids: List[int] = []
        for p in paths_raw:
            path_tuple = tuple(p)
            if path_tuple not in self.path_to_id:
                raise KeyError(f"Path {path_tuple} not found in path_to_id.")
            path_ids.append(self.path_to_id[path_tuple])

        if len(path_ids) == 0:
            raise ValueError(f"No valid label path for example index {idx}.")

        # Compute label_ids for BCE
        label_ids: List[int] = []
        if self.label_key in ex and ex[self.label_key]:
            labs = ex[self.label_key]
            if isinstance(labs, (list, tuple)):
                for lab in labs:
                    if isinstance(lab, str) and lab in self.label_to_id:
                        label_ids.append(self.label_to_id[lab])
        else:
            for p in paths_raw:
                for node in p:
                    if node in self.label_to_id:
                        label_ids.append(self.label_to_id[node])
        label_ids = sorted(set(label_ids))
        if len(label_ids) == 0:
            raise ValueError(f"No valid label node for example index {idx}.")

        # ---- Collect ALL path descriptions for this example (multi-positive) ----
        desc_texts: List[str] = []
        if self.desc_key in ex and ex[self.desc_key]:
            raw_desc = ex[self.desc_key]
            if isinstance(raw_desc, list):
                desc_texts = [d for d in raw_desc if isinstance(d, str) and d.strip() != ""]
            elif isinstance(raw_desc, str) and raw_desc.strip() != "":
                desc_texts = [raw_desc.strip()]
            else:
                desc_texts = []
        else:
            desc_texts = []

        # Tokenize each description separately; keep a list of [L] tensors.
        desc_input_ids_list: List[torch.Tensor] = []
        desc_attention_mask_list: List[torch.Tensor] = []
        for d in desc_texts:
            desc_enc = self.tokenizer(
                d,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,  # can use a separate desc_max_length later
                return_tensors="pt",
            )
            desc_input_ids_list.append(desc_enc["input_ids"].squeeze(0))
            desc_attention_mask_list.append(desc_enc["attention_mask"].squeeze(0))

        # Backward-compatible single-desc tensors (first desc if exists, else zeros)
        if len(desc_input_ids_list) > 0:
            desc_input_ids = desc_input_ids_list[0]
            desc_attention_mask = desc_attention_mask_list[0]
        else:
            desc_input_ids = torch.zeros_like(input_ids)
            desc_attention_mask = torch.zeros_like(attention_mask)

        # if idx < 5:
        #     print("\n===== DATA DEBUG =====")
        #     print("text:", text[:120])
        #     print("raw label:", ex.get(self.label_key, None))
        #     print("raw path:", ex.get(self.path_key, None))
        #     print("path_ids:", path_ids)
        #     print("label_ids:", label_ids)
        #     print("num_paths:", len(path_ids), "num_labels:", len(label_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "path_ids": path_ids,
            "label_ids": label_ids,
            "raw": ex,
            "desc_input_ids": desc_input_ids,
            "desc_attention_mask": desc_attention_mask,
            "desc_input_ids_list": desc_input_ids_list,
            "desc_attention_mask_list": desc_attention_mask_list,
        }


def hierarchical_collate_fn(batch: List[Dict[str, Any]], num_labels: int) -> Dict[str, Any]:
    """
    Collate function for HierarchicalTextDataset.

    Produces:
        - input_ids:         [B, L]
        - attention_mask:    [B, L]
        - batch_paths:       List[List[int]]  for HierarchyContrastiveLoss
        - labels_multi_hot:  [B, P] multi-hot tensor for BCE
        - raw:               List[raw_example_dict]
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    batch_paths = [item["path_ids"] for item in batch]
    batch_label_ids = [item["label_ids"] for item in batch]
    raw_examples = [item["raw"] for item in batch]

    # Flatten all descriptions in the batch (multi-positive). Also store which example each desc belongs to.
    desc_input_ids_flat_list: List[torch.Tensor] = []
    desc_attention_mask_flat_list: List[torch.Tensor] = []
    desc_owner: List[int] = []
    for b_idx, item in enumerate(batch):
        d_ids_list = item.get("desc_input_ids_list", [])
        d_mask_list = item.get("desc_attention_mask_list", [])
        for di, dm in zip(d_ids_list, d_mask_list):
            desc_input_ids_flat_list.append(di)
            desc_attention_mask_flat_list.append(dm)
            desc_owner.append(b_idx)

    if len(desc_input_ids_flat_list) > 0:
        desc_input_ids = torch.stack(desc_input_ids_flat_list, dim=0)          # [M, L]
        desc_attention_mask = torch.stack(desc_attention_mask_flat_list, dim=0) # [M, L]
        desc_owner = torch.tensor(desc_owner, dtype=torch.long)                 # [M]
    else:
        # No descriptions in this batch; create empty tensors.
        desc_input_ids = torch.zeros((0, input_ids.size(1)), dtype=input_ids.dtype)
        desc_attention_mask = torch.zeros((0, attention_mask.size(1)), dtype=attention_mask.dtype)
        desc_owner = torch.zeros((0,), dtype=torch.long)

    B = input_ids.size(0)
    P = num_labels
    labels_multi_hot = torch.zeros(B, P, dtype=torch.float32)
    for b, label_ids in enumerate(batch_label_ids):
        for lid in label_ids:
            labels_multi_hot[b, lid] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "batch_paths": batch_paths,
        "labels": labels_multi_hot,
        "raw": raw_examples,
        "desc_input_ids": desc_input_ids,
        "desc_attention_mask": desc_attention_mask,
        "desc_owner": desc_owner,
    }