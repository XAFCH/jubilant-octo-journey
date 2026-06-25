from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel


class ContrastiveClassifier(nn.Module):
    def __init__(self, model_name, num_labels, embedding_dim=768, projection_dim=128):
        super(ContrastiveClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.classification_head = nn.Linear(embedding_dim, num_labels)

    def forward(self, text_input_ids, text_attention_mask, description_input_ids, description_attention_mask):
        # Encode text
        text_outputs = self.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Encode description
        description_outputs = self.encoder(
            input_ids=description_input_ids,
            attention_mask=description_attention_mask
        )
        description_embeddings = description_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Projection for contrastive loss
        text_projection = self.projection_head(text_embeddings)
        description_projection = self.projection_head(description_embeddings)

        # Classification logits
        logits = self.classification_head(text_embeddings)

        return text_projection, description_projection, logits, text_embeddings


def contrastive_loss(text_proj, desc_proj, temperature=0.07):
    batch_size = text_proj.size(0)

    # Normalize the projections
    text_proj_norm = nn.functional.normalize(text_proj, dim=1)
    desc_proj_norm = nn.functional.normalize(desc_proj, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(text_proj_norm, desc_proj_norm.T)  # Shape: (batch_size, batch_size)

    # Create labels: diagonal elements are positives
    labels = torch.arange(batch_size).to(text_proj.device)

    # Scale similarities by temperature
    similarity_matrix = similarity_matrix / temperature

    # Use cross-entropy loss
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

    return loss


class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name, num_labels, embedding_dim=768, projection_dim=128):
        super(HierarchicalClassifier, self).__init__()

        # Text encoder (e.g., BERT)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_labels)

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.1)))

    def forward(self, text_input_ids, text_attention_mask):
        # Encode text
        outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        return logits, pooled_output  # Return logits and embeddings

    def encode(self, text_input_ids, text_attention_mask):
        # Get embeddings
        outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Apply projection head
        projected_output = self.projection_head(pooled_output)
        return projected_output  # Return projected embeddings

    def encode_positives(self, positive_input_ids_list, positive_attention_mask_list):
        """
        Encodes a list of positive examples for each anchor.

        Parameters:
        - positive_input_ids_list: List of tensors [num_positives_i, seq_len]
        - positive_attention_mask_list: List of tensors [num_positives_i, seq_len]

        Returns:
        - positives_embeddings_list: List of tensors [num_positives_i, projection_dim]
        """
        positives_embeddings_list = []
        for input_ids, attention_mask in zip(positive_input_ids_list, positive_attention_mask_list):
            if input_ids.size(0) == 0:
                # No positive examples
                positives_embeddings_list.append(
                    torch.empty(0, self.projection_head[-1].out_features).to(input_ids.device))
                continue

            # Encode positives
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output  # [num_positives_i, hidden_size]
            pos_embeddings = self.projection_head(pooled_output)  # [num_positives_i, projection_dim]
            positives_embeddings_list.append(pos_embeddings)
        return positives_embeddings_list

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)  # ensure temperature is always positive


class HierarchicalMultiLabelModel(nn.Module):
    """
    Multi-label text classification model with a transformer encoder and
    a label-path embedding matrix tied to the classifier weights.

    - Encoder: HuggingFace transformer (e.g., BERT).
    - Text embedding: use [CLS] (first token) embedding.
    - Classifier: linear layer producing logits over all label paths.
    - Path embeddings for contrastive loss: classifier weight matrix.

    Args:
        encoder_name:   HuggingFace model name or path, e.g. "bert-base-uncased".
        num_labels:     Total number of label paths (P) in the global taxonomy.
        dropout:        Dropout probability before the classifier.
        use_cls_token:  If True, use hidden_state[:, 0, :] as text embedding.
                        If False, use mean pooled hidden states.

    Forward inputs:
        input_ids:      [B, L] token ids
        attention_mask: [B, L] attention mask
        desc_input_ids: [M, L] (flattened description batch; M may differ from B and may be 0)
        desc_attention_mask: [M, L]
        desc_token_type_ids: [M, L] (optional)

    Forward outputs (dict):
        logits:         [B, P] classification logits (before sigmoid)
        text_embs:      [B, D] text embeddings
        path_embs:      [P, D] path embeddings (same D, from classifier weights)
        text_proj:      [B, projection_dim] text projections
        desc_embs:      [B, D] or None description embeddings
        desc_proj:      [M, projection_dim] or None description projections
    """

    def __init__(
            self,
            encoder_name: str,
            num_labels: int,  # label nodes (for BCE)
            num_paths: int,  # paths (for HCL)
            dropout: float = 0.1,
            use_cls_token: bool = True,
            projection_dim: int = 128,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_labels = num_labels
        self.num_paths = num_paths
        self.use_cls_token = use_cls_token

        self.dropout = nn.Dropout(dropout)

        # classifier weight shape: [num_labels, hidden_size]
        # we will treat each row as the embedding of a label path
        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=True)
        # optional: init classifier weights a bit smaller
        nn.init.xavier_uniform_(self.classifier.weight)

        # Embedding over PATHS (free parameters for HCL)
        self.path_embedding = nn.Embedding(self.num_paths, self.hidden_size)

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.hcl_text_head = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.hcl_path_head = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )


    def _pool_text(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states into a single vector per example.

        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L]

        Returns:
            pooled: [B, D]
        """
        if self.use_cls_token:
            # [CLS] / first token as representation
            # many HuggingFace models use token 0 as CLS.
            pooled = hidden_states[:, 0, :]
        else:
            # mean pooling over non-masked tokens
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            masked_hidden = hidden_states * mask
            sums = masked_hidden.sum(dim=1)  # [B, D]
            lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            pooled = sums / lengths
        return pooled

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            desc_input_ids: Optional[torch.Tensor] = None,
            desc_attention_mask: Optional[torch.Tensor] = None,
            desc_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]
            token_type_ids: [B, L] (optional, for models like BERT)
            desc_input_ids: [M, L] (M may be 0)
            desc_attention_mask: [M, L]
            desc_token_type_ids: [M, L] (optional, for models like BERT)

        Returns:
            dict with:
                logits:     [B, P]
                text_embs:  [B, D]
                path_embs:  [P, D]
                text_proj:  [B, projection_dim]
                desc_embs:  [B, D] or None
                desc_proj:  [M, projection_dim] or None
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # encoder_outputs.last_hidden_state: [B, L, D]
        hidden_states = encoder_outputs.last_hidden_state

        # pool to [B, D]
        text_embs = self._pool_text(hidden_states, attention_mask)
        text_hcl = self.hcl_text_head(text_embs)
        text_embs = self.dropout(text_embs)

        text_proj = self.projection_head(text_embs)

        desc_embs = None
        desc_proj = None
        # desc inputs are a flattened batch with shape [M, L], M may differ from B and may be 0
        if desc_input_ids is not None and desc_attention_mask is not None and desc_input_ids.size(0) > 0:
            desc_outputs = self.encoder(
                input_ids=desc_input_ids,
                attention_mask=desc_attention_mask,
                token_type_ids=desc_token_type_ids
            )
            desc_hidden_states = desc_outputs.last_hidden_state
            desc_embs = self._pool_text(desc_hidden_states, desc_attention_mask)
            desc_embs = self.dropout(desc_embs)
            desc_proj = self.projection_head(desc_embs)

        # classification logits: [B, P]
        logits = self.classifier(text_embs)

        path_embs = self.path_embedding.weight  # [num_paths, hidden_size]
        path_hcl = self.hcl_path_head(path_embs)

        return {
            "logits": logits,
            "text_embs": text_embs,
            "path_embs": path_embs,  # [num_paths, D]
            "text_proj": text_proj,
            "desc_embs": desc_embs,
            "desc_proj": desc_proj,
            "text_hcl": text_hcl,
            "path_hcl": path_hcl,
        }