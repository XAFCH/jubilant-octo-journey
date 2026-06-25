from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label focal loss on logits.

    This follows the standard sigmoid-based multi-label focal form:
        FL = alpha_t * (1 - p_t)^gamma * BCEWithLogits
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction={self.reduction}.")


class AsymmetricLossMultiLabel(nn.Module):
    """
    Multi-label Asymmetric Loss (ASL) adapted from Alibaba-MIIL/ASL.

    The original repo sums the loss tensor. Here we expose a reduction option and
    default to mean so it can act as a drop-in replacement for BCE in this codebase.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float | None = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        anti_targets = 1.0 - targets

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            pt = xs_pos * targets + xs_neg * anti_targets
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * anti_targets
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss = loss * one_sided_w

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction={self.reduction}.")


def build_multilabel_loss(
    loss_name: str,
    *,
    bce_pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.25,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float | None = 0.05,
    asl_eps: float = 1e-8,
    asl_disable_grad: bool = False,
) -> nn.Module:
    name = loss_name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
    if name == "focal":
        return MultiLabelFocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            reduction="mean",
        )
    if name == "asl":
        return AsymmetricLossMultiLabel(
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
            eps=asl_eps,
            disable_torch_grad_focal_loss=asl_disable_grad,
            reduction="mean",
        )
    raise ValueError(f"Unknown multilabel loss '{loss_name}'. Expected one of: bce, focal, asl.")
