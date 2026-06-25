from __future__ import annotations
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from hierarchy_from_taxonomy import prepare_hierarchy_from_taxonomy
from datasets import HierarchicalTextDataset, hierarchical_collate_fn
from model import HierarchicalMultiLabelModel
from hierarchical_contrastive_loss import HierarchyContrastiveLoss
from eval import evaluate_multilabel

def make_dataloader(
    data_path: str,
    tokenizer,
    path_to_id: Dict,
    num_labels: int,
    batch_size: int,
    max_length: int,
    shuffle: bool = True,
):
    ds = HierarchicalTextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        num_labels=num_labels,
        text_key="token",
        path_key="path",   # assumes your JSON has "path": [[...], ...]
        label_key="label", # fallback
        max_length=max_length,
    )

    collate = lambda batch: hierarchical_collate_fn(batch, num_labels=num_labels)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )
    return loader


def train_one_epoch(
    model: nn.Module,
    hcl: HierarchyContrastiveLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LinearSchedulerWithWarmup = None,
):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_hcl = 0.0
    steps = 0

    bce_loss_fn = nn.BCEWithLogitsLoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_paths = batch["batch_paths"]              # List[List[int]]
        labels = batch["labels"].to(device)             # [B, P]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]        # [B, P]
        text_embs = outputs["text_embs"]  # [B, D]
        path_embs = outputs["path_embs"]  # [P, D]

        bce_loss = bce_loss_fn(logits, labels)
        h_loss = hcl(text_embs, path_embs, batch_paths)

        loss = bce_loss + beta * h_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() 

        steps += 1
        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_hcl += h_loss.item()

    return {
        "loss": total_loss / steps,
        "bce": total_bce / steps,
        "hcl": total_hcl / steps,
    }


def main():
    # ======== config ========
    encoder_name = "bert-base-uncased"
    taxonomy_path = "bgc.taxonomy"   # same taxonomy structure shared
    labels_path   = "labels.txt"     # global label vocab
    train_path    = "bgc_train.jsonl"  # change for WoS/NYT
    batch_size    = 8
    max_length    = 256
    epochs        = 3
    lr            = 2e-5
    beta          = 1.0   # weight for hierarchy contrastive loss
    tau           = 0.1   # temperature for contrastive loss
    lam           = 0.5   # lambda in exp(-lam * d) for w_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== hierarchy: paths, path_to_id, w_matrix ========
    paths, path_to_id, w_matrix = prepare_hierarchy_from_taxonomy(
        taxonomy_path=taxonomy_path,
        labels_path=labels_path,
        root_name="Root",
        lam=lam,
    )

    num_labels = len(paths)
    print(f"Num label paths: {num_labels}")

    # ======== tokenizer & dataloader ========
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    train_loader = make_dataloader(
        data_path=train_path,
        tokenizer=tokenizer,
        path_to_id=path_to_id,
        num_labels=num_labels,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )

    # ======== model & losses ========
    model = HierarchicalMultiLabelModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
        dropout=0.1,
        use_cls_token=True,
    ).to(device)

    hcl = HierarchyContrastiveLoss(
        w_matrix=w_matrix.to(device),
        tau=tau,
        reduction="mean",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Number of training steps
    num_training_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warm-up

    # Warmup: 10% of steps (you can tune this)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


    patience = 3       # how many epochs with no improvement before stopping
    best_micro_f1 = 0.0
    epochs_no_improve = 0

    # ======== training loop ========
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(
            model=model,
            hcl=hcl,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            beta=beta,
        )
        print(
            f"Epoch {epoch}: "
            f"loss={stats['loss']:.4f} "
            f"(bce={stats['bce']:.4f}, hcl={stats['hcl']:.4f})"
        )

            # ---- Validation ----
        val_stats = evaluate_multilabel(
            model=model,
            loader=train_loader,
            device=device,
            threshold=0.5,
        )

        print(
        f"Val: micro_P={val_stats['micro_precision']:.4f} "
        f"micro_R={val_stats['micro_recall']:.4f} "
        f"micro_F1={val_stats['micro_f1']:.4f} "
        f"macro_F1={val_stats['macro_f1']:.4f}"
        )

        # ---- Early stopping on micro-F1 ----
        current_micro_f1 = val_stats["micro_f1"]

        if current_micro_f1 > best_micro_f1 + 1e-6:  # small tolerance
            best_micro_f1 = current_micro_f1
            epochs_no_improve = 0

            # Save best model so far
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"New best micro-F1: {best_micro_f1:.4f} (model saved)")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in micro-F1 for {patience} epochs.")
            break

    # You can save the model if you like:
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/hier_model_bgc.pt")
    print("Model saved to checkpoints/hier_model_bgc.pt")

    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.to(device)
    test_stats = evaluate_multilabel(model, train_loader, device=device, threshold=0.5)
    print("Test:", test_stats)

if __name__ == "__main__":
    main()