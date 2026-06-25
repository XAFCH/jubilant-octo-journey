import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from model import ContrastiveClassifier
from hierarchical_BCE import hierarchical_bce_loss_with_penalty
from hierarchical_contrastive_loss import compute_max_depth, get_label_path, remove_subpaths, compute_hierarchical_similarity, hierarchical_contrastive_loss
from focal_loss import HierarchicalFocalLossWithPenalty

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

# Load the dataset with descriptions and label list:
f = open('data/WOS/wos_debug_train.json', 'r')
train_data = f.readlines()
f.close()
structured_train = []
for item in train_data:
    item = json.loads(item)
    structured_train.append(item)

f = open('data/WOS/wos_debug_test.json', 'r')
test_data = f.readlines()
f.close()
structured_test = []
for item in test_data:
    item = json.loads(item)
    structured_test.append(item)

# Create hierarchy mapping (label index to parent label index)
# hierarchy = {
#     'Level2_A1': 'Level1_A',
#     'Level2_A2': 'Level1_A',
#     'Level1_A': 'Root',
#     'Level1_B': 'Root',
#     'Root': None
# }
hierarchy = {}
with open('data/WOS/wos.taxonomy', 'r') as file:
    for line in file:
        # Split the line by tab to separate labels
        labels = line.strip().split('\t')
        
        # The first label is the parent, the rest are child labels
        parent_label = labels[0]
        child_labels = labels[1:]
        
        # Create child-parent relationships
        for child_label in child_labels:
            hierarchy[child_label] = parent_label
hierarchy['Root'] = None

with open('data/WOS/labels.txt', 'r') as file:
    labels = [line.strip() for line in file]
# Create label to index mapping
label_to_index = {label: idx for idx, label in enumerate(sorted(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}

hierarchy_index = {}
# Convert the hierachy to use indices
for child_label, parent_label in hierarchy.items():
    parent_index = label_to_index.get(parent_label)
    child_index = label_to_index.get(child_label)

    if parent_index is not None and child_index is not None:
        hierarchy_index[child_index] = parent_index

max_depth = compute_max_depth(hierarchy_index)

mlb = MultiLabelBinarizer(classes=labels)
mlb.fit(labels)

# Initialize the Pre-trained LLM and Tokenizer
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = AutoModel.from_pretrained(model_name)

# Create a Custom Dataset
class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        """
        data: a list of dictionaries with keys 'anchor' and 'positive'
        tokenizer: the tokenizer for BERT/RoBERTa
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        anchor_text = sample['abstract']
        anchor_text = clean_str(anchor_text)
        positive_text = sample['path_description']
        positive_text = clean_str(positive_text)
        labels = sample['label']
        label_indices = [label_to_index[label] for label in labels]
        # print("label index: ", label_indices)
        # label_indices = torch.tensor(label_indices, dtype=torch.long)
            
        # Tokenize texts
        anchor_inputs = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        positive_inputs = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert labels to binary vector using MultiLabelBinarizer
        label_vector = mlb.transform([labels])[0]  # Transform labels and take the first row
        label_tensor = torch.tensor(label_vector, dtype=torch.float)  # Convert to tensor

            
        return {
            'anchor_input_ids': anchor_inputs['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_inputs['attention_mask'].squeeze(),
            'positive_input_ids': positive_inputs['input_ids'].squeeze(),
            'positive_attention_mask': positive_inputs['attention_mask'].squeeze(),
            'label': label_tensor,
            'label_indices': label_indices
        }

def custom_collate_fn(batch):
    # Collate individual components separately
    anchor_input_ids = torch.stack([item['anchor_input_ids'] for item in batch])
    anchor_attention_mask = torch.stack([item['anchor_attention_mask'] for item in batch])
    positive_input_ids = torch.stack([item['positive_input_ids'] for item in batch])
    positive_attention_mask = torch.stack([item['positive_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # For label_indices, keep as a list of tensors
    label_indices = [item['label_indices'] for item in batch]  # This will be a list of tensors
    
    return {
        'anchor_input_ids': anchor_input_ids,
        'anchor_attention_mask': anchor_attention_mask,
        'positive_input_ids': positive_input_ids,
        'positive_attention_mask': positive_attention_mask,
        'label': labels,
        'label_indices': label_indices  # Keep label_indices as a list of tensors
    }

# Load and Preprocess Data:
batch_size = 16

train_dataset = ContrastiveDataset(structured_train, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

eval_dataset = ContrastiveDataset(structured_test, tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

test_dataset = ContrastiveDataset(structured_test, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

# Define the Model and Loss Function
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
   
# Training Loop with Fine-Tuning
classification_criterion = nn.BCEWithLogitsLoss()

num_labels = len(labels)
model = ContrastiveClassifier('bert-base-uncased', num_labels)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

print('start training!')
# Training loop
num_epochs = 10
temperature = 0.07
lambda_contrastive = 0.5
lambda_classification = 2.0
lambda_hier = 0.5

focal_fn = HierarchicalFocalLossWithPenalty(
            hierarchy=hierarchy,
            gamma=2.0,
            lambda_hier=0.5,
            smoothing=0.1,
            reduction='mean'
        )

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to device
        text_input_ids = batch['anchor_input_ids'].to(device)
        text_attention_mask = batch['anchor_attention_mask'].to(device)
        description_input_ids = batch['positive_input_ids'].to(device)
        description_attention_mask = batch['positive_attention_mask'].to(device)
        labels = batch['label'].to(device)
        anchor_labels = batch['label_indices']  # List of labels per anchor (list of lists)
        # print("number of anchor: ", anchor_labels)

        # Generate and filter label paths for anchors
        anchor_filtered_paths_batch = []
        for labels_per_anchor in anchor_labels:
            label_paths = [get_label_path(label, hierarchy_index) for label in labels_per_anchor]
            filtered_paths = remove_subpaths(label_paths)
            # print("filter path", filtered_paths)
            anchor_filtered_paths_batch.append(filtered_paths)
        # print("number of filtered path:", len(anchor_filtered_paths_batch))

        # Forward pass
        text_proj, desc_proj, logits, text_embeddings = model(
            text_input_ids,
            text_attention_mask,
            description_input_ids,
            description_attention_mask
        )

        # Prepare negatives
        batch_size = len(anchor_filtered_paths_batch)
        negative_descriptions = []
        negative_filtered_paths_batch = []

        for i in range(batch_size):
            mask = torch.ones(batch_size, dtype=torch.bool)
            mask[i] = False  # Exclude the anchor's own positive
            negatives_desc_i = desc_proj[mask].to(device)  # [num_negatives, embedding_dim]
            negatives_labels_i = [anchor_labels[j] for j in range(batch_size) if j != i]

            # Generate and filter label paths for negatives
            negatives_filtered_paths_i = []
            for labels_per_negative in negatives_labels_i:
                label_paths_neg = [get_label_path(label, hierarchy_index) for label in labels_per_negative]
                filtered_paths_neg = remove_subpaths(label_paths_neg)
                negatives_filtered_paths_i.append(filtered_paths_neg)

            negative_descriptions.append(negatives_desc_i)
            negative_filtered_paths_batch.append(negatives_filtered_paths_i)

        # Stack negatives
        negative_descriptions = torch.stack(negative_descriptions)  # [batch_size, num_negatives, embedding_dim]

        # Compute hierarchical similarities
        weights = torch.zeros(batch_size, batch_size - 1).to(device)
        for i in range(batch_size):
            anchor_paths = anchor_filtered_paths_batch[i]
            for j in range(batch_size - 1):
                negative_paths = negative_filtered_paths_batch[i][j]
                sim_score = compute_hierarchical_similarity(anchor_paths, negative_paths, max_depth)
                weights[i, j] = sim_score

        # Compute hierarchical contrastive loss
        loss_contrastive = hierarchical_contrastive_loss(
            text_proj, desc_proj, negative_descriptions, weights, temperature
        )

        # Compute hierarchical BCE loss
        # loss_classification = hierarchical_bce_loss_with_penalty(
        #     logits, labels, hierarchy, lambda_hier
        # )

        # Compute focal loss 
        loss_focal = focal_fn(logits, labels)

        # Total loss and backpropagation
        # total_loss_batch = lambda_contrastive * loss_contrastive + lambda_classification * loss_classification
        total_loss_batch = lambda_contrastive * loss_contrastive + lambda_classification * loss_focal
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # model.eval()

    # eval_all_labels = []
    # eval_all_preds = []
    # with torch.no_grad():
    #     for batch in eval_dataloader:
    #         text_input_ids = batch['anchor_input_ids'].to(device)
    #         text_attention_mask = batch['anchor_attention_mask'].to(device)
    #         labels = batch['label'].cpu().numpy()

    #         # Forward pass
    #         outputs = model.encoder(
    #             input_ids=text_input_ids,
    #             attention_mask=text_attention_mask
    #         )
        
    #         text_embeddings = outputs.last_hidden_state[:, 0, :]
    #         logits = model.classification_head(text_embeddings)
    #         preds = torch.sigmoid(logits).cpu().numpy()

    #         eval_all_labels.append(labels)
    #         eval_all_preds.append(preds)
    
    # # Concatenate results
    # all_labels = np.vstack(all_labels)
    # all_preds = np.vstack(all_preds)

    # # Binarize predictions with a threshold
    # threshold = 0.5
    # all_preds_bin = (all_preds >= threshold).astype(int)

    # f1_micro = f1_score(all_labels, all_preds_bin, average='micro')

    # # Early Stopping


print('start evaluation!')
# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        text_input_ids = batch['anchor_input_ids'].to(device)
        text_attention_mask = batch['anchor_attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()

        # Forward pass
        outputs = model.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        logits = model.classification_head(text_embeddings)
        preds = torch.sigmoid(logits).cpu().numpy()

        all_labels.append(labels)
        all_preds.append(preds)

# Concatenate results
all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)

# Binarize predictions with a threshold
threshold = 0.5
all_preds_bin_5 = (all_preds >= threshold).astype(int)

# Compute metrics
f1_micro = f1_score(all_labels, all_preds_bin_5, average='micro')
precision_micro = precision_score(all_labels, all_preds_bin_5, average='micro')
recall_micro = recall_score(all_labels, all_preds_bin_5, average='micro')
# hamming = hamming_loss(all_labels, all_preds_bin)

f1_macro = f1_score(all_labels, all_preds_bin_5, average='macro')
precision_macro = precision_score(all_labels, all_preds_bin_5, average='macro')
recall_macro = recall_score(all_labels, all_preds_bin_5, average='macro')

print(f"F1 Score (Micro): {f1_micro:.4f}")
print(f"Precision (Micro): {precision_micro:.4f}")
print(f"Recall (Micro): {recall_micro:.4f}")
# print(f"Hamming Loss: {hamming:.4f}")

print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")


# Binarize predictions with a threshold
threshold = 0.3
all_preds_bin_3 = (all_preds >= threshold).astype(int)

# Compute metrics
f1_micro_3 = f1_score(all_labels, all_preds_bin_3, average='micro')
precision_micro_3 = precision_score(all_labels, all_preds_bin_3, average='micro')
recall_micro_3 = recall_score(all_labels, all_preds_bin_3, average='micro')
# hamming = hamming_loss(all_labels, all_preds_bin)

f1_macro_3 = f1_score(all_labels, all_preds_bin_3, average='macro')
precision_macro_3 = precision_score(all_labels, all_preds_bin_3, average='macro')
recall_macro_3 = recall_score(all_labels, all_preds_bin_3, average='macro')

print(f"F1 Score (Micro): {f1_micro_3:.4f}")
print(f"Precision (Micro): {precision_micro_3:.4f}")
print(f"Recall (Micro): {recall_micro_3:.4f}")
# print(f"Hamming Loss: {hamming:.4f}")

print(f"F1 Score (Macro): {f1_macro_3:.4f}")
print(f"Precision (Macro): {precision_macro_3:.4f}")
print(f"Recall (Macro): {recall_macro_3:.4f}")


# Binarize predictions with a threshold
threshold = 0.25
all_preds_bin_1 = (all_preds >= threshold).astype(int)

# Compute metrics
f1_micro_1 = f1_score(all_labels, all_preds_bin_1, average='micro')
precision_micro_1 = precision_score(all_labels, all_preds_bin_1, average='micro')
recall_micro_1 = recall_score(all_labels, all_preds_bin_1, average='micro')
# hamming = hamming_loss(all_labels, all_preds_bin)

f1_macro_1 = f1_score(all_labels, all_preds_bin_1, average='macro')
precision_macro_1 = precision_score(all_labels, all_preds_bin_1, average='macro')
recall_macro_1 = recall_score(all_labels, all_preds_bin_1, average='macro')

print(f"F1 Score (Micro): {f1_micro_1:.4f}")
print(f"Precision (Micro): {precision_micro_1:.4f}")
print(f"Recall (Micro): {recall_micro_1:.4f}")
# print(f"Hamming Loss: {hamming:.4f}")

print(f"F1 Score (Macro): {f1_macro_1:.4f}")
print(f"Precision (Macro): {precision_macro_1:.4f}")
print(f"Recall (Macro): {recall_macro_1:.4f}")


# Binarize predictions with a threshold
threshold = 0.2
all_preds_bin_05 = (all_preds >= threshold).astype(int)

# Compute metrics
f1_micro_05 = f1_score(all_labels, all_preds_bin_05, average='micro')
precision_micro_05 = precision_score(all_labels, all_preds_bin_05, average='micro')
recall_micro_05 = recall_score(all_labels, all_preds_bin_05, average='micro')
# hamming = hamming_loss(all_labels, all_preds_bin)

f1_macro_05 = f1_score(all_labels, all_preds_bin_05, average='macro')
precision_macro_05 = precision_score(all_labels, all_preds_bin_05, average='macro')
recall_macro_05 = recall_score(all_labels, all_preds_bin_05, average='macro')

print(f"F1 Score (Micro): {f1_micro_05:.4f}")
print(f"Precision (Micro): {precision_micro_05:.4f}")
print(f"Recall (Micro): {recall_micro_05:.4f}")
# print(f"Hamming Loss: {hamming:.4f}")

print(f"F1 Score (Macro): {f1_macro_05:.4f}")
print(f"Precision (Macro): {precision_macro_05:.4f}")
print(f"Recall (Macro): {recall_macro_05:.4f}")