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
from model import ContrastiveClassifier, contrastive_loss

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

with open('data/WOS/labels.txt', 'r') as file:
    labels = [line.strip() for line in file]
mlb = MultiLabelBinarizer(classes=labels)
mlb.fit(labels)

# Initialize the Pre-trained LLM and Tokenizer
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = AutoModel.from_pretrained(model_name)

# Create a Custom Dataset
class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
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
        # print('anchor_text: ', anchor_text)
        positive_text = sample['path_description']
        # print('positve_text: ', positive_text)
        positive_text = clean_str(positive_text)
        # print('positve_text: ', positive_text)
        labels = sample['label']
            
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
            'label': label_tensor
        }

# Load and Preprocess Data:
batch_size = 16

train_dataset = ContrastiveDataset(structured_train, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ContrastiveDataset(structured_test, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the Model and Loss Function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
# Training Loop with Fine-Tuning
classification_criterion = nn.BCEWithLogitsLoss()

num_labels = len(labels)
model = ContrastiveClassifier('bert-base-uncased', num_labels)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


num_epochs = 5
temperature = 0.07
lambda_contrastive = 1.0
lambda_classification = 1.0

print('start training!')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        text_input_ids = batch['anchor_input_ids'].to(device)
        text_attention_mask = batch['anchor_attention_mask'].to(device)
        description_input_ids = batch['positive_input_ids'].to(device)
        description_attention_mask = batch['positive_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        text_proj, desc_proj, logits, text_embeddings = model(
            text_input_ids,
            text_attention_mask,
            description_input_ids,
            description_attention_mask
        )

        # Compute losses
        loss_contrastive = contrastive_loss(text_proj, desc_proj, temperature)
        loss_classification = classification_criterion(logits, labels)

        total_batch_loss = lambda_contrastive * loss_contrastive + lambda_classification * loss_classification

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

print('start evaluation!')
# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        text_input_ids = batch['anchor_input_ids'].to(device)
        text_attention_mask = batch['anchor_attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()

        # Forward pass
        text_outputs = model.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        logits = model.classification_head(text_embeddings)
        preds = torch.sigmoid(logits).cpu().numpy()

        all_labels.append(labels)
        all_preds.append(preds)

# Concatenate results
all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)

# Binarize predictions with a threshold
threshold = 0.1
all_preds_bin = (all_preds >= threshold).astype(int)

# Compute metrics
f1_micro = f1_score(all_labels, all_preds_bin, average='micro')
precision_micro = precision_score(all_labels, all_preds_bin, average='micro')
recall_micro = recall_score(all_labels, all_preds_bin, average='micro')

f1_macro = f1_score(all_labels, all_preds_bin, average='macro')
precision_macro = precision_score(all_labels, all_preds_bin, average='macro')
recall_macro = recall_score(all_labels, all_preds_bin, average='macro')
# hamming = hamming_loss(all_labels, all_preds_bin)

print(f"F1 Score (Micro): {f1_micro:.4f}")
print(f"Precision (Micro): {precision_micro:.4f}")
print(f"Recall (Micro): {recall_micro:.4f}")
# print(f"Hamming Loss: {hamming:.4f}")

print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")