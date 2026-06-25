import argparse
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import unicodedata
from model import HierarchicalClassifier
from hierarchical_BCE import hierarchical_bce_loss_with_penalty
from hierarchical_infoNCE import get_label_path, compute_negative_weights, hierarchical_info_nce_loss
from infoNCE import hierarchical_info_nce_loss
from focal_loss import HierarchicalFocalLossWithPenalty


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def normalize_text(text):
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Replace special quotation marks and dashes with standard ones
    text = text.translate({
        ord('‘'): ord("'"),
        ord('’'): ord("'"),
        ord('“'): ord('"'),
        ord('”'): ord('"'),
        ord('—'): ord('-'),
        ord('–'): ord('-'),
    })

    # Remove control characters and non-printable characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    # Optionally, handle URLs and email addresses
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def clean_str(string):
    string = re.sub(r'http\S+|www.\S+', '[URL]', string)
    string = re.sub(r'\S+@\S+', '[EMAIL]', string)
    return string.strip()

def compute_max_depth(hierarchy):
    max_depth = 0
    for label in hierarchy.keys():
        path = get_label_path(label, hierarchy)
        if len(path) > max_depth:
            max_depth = len(path)
    return max_depth

# Data structure 
# a list of dictionaries with keys "abstract", "labels", "path", "positives", where "abstract" is the original text of the anchor, and "labels" is a list of label that contains golden truth labels
# "path" is a list of list, each element contain a path that represent  
# the true labels; "positives" is a list of list, each element in the list is the generated description of the path. 

# Create a Custom Dataset
class MultipathDataset(Dataset):
    def __init__(self, data, tokenizer, dataset, max_length=256):
        """
        data: a list of dictionaries with keys 'abstract' and 'labels', 'positives', 'paths'
        tokenizer: the tokenizer for BERT/RoBERTa
        """
        # self.data = data
        self.data = [sample for sample in data if sample['path_description']] # remove the example if the path description is empty 
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        anchor_text = sample['token']
        if self.dataset == 'BGC':
            anchor_text = normalize_text(anchor_text)
        anchor_text = clean_str(anchor_text)
        positive_text = sample['path_description']
        labels = sample['label']
        label_indices = [label_to_index[label] for label in labels]
        paths = []
        for path in sample['path']:
            path_idx = []
            for label in path:
                label_idx = label_to_index[label]
                path_idx.append(label_idx)
            paths.append(path_idx)
            
        # Tokenize texts
        anchor_inputs = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        positive_encodings = []
        for path, positive in zip(paths, positive_text):
            positive = clean_str(positive)
            encoding = self.tokenizer(
                positive,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            positive_encodings.append({'encodings': encoding,
                                       'path': path
                                       })

        # Convert labels to binary vector using MultiLabelBinarizer
        label_vector = mlb.transform([labels])[0]  # Transform labels and take the first row
        label_tensor = torch.tensor(label_vector, dtype=torch.float)  # Convert to tensor
 
        return {
            'anchor_input_ids': anchor_inputs['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_inputs['attention_mask'].squeeze(),
            'positive_encodings': positive_encodings,
            'label': label_tensor,
            'label_indices': label_indices,
            'paths': paths
        }

def custom_collate_fn(batch):
    # Collate individual components separately
    anchor_input_ids = torch.stack([item['anchor_input_ids'] for item in batch])
    anchor_attention_mask = torch.stack([item['anchor_attention_mask'] for item in batch])
    # positive_input_ids = torch.stack([item['positive_input_ids'] for item in batch])
    # positive_attention_mask = torch.stack([item['positive_attention_mask'] for item in batch])
    
    labels = torch.stack([item['label'] for item in batch])
    
    # For label_indices, keep as a list of tensors
    label_indices = [item['label_indices'] for item in batch]  # This will be a list of tensors


    positive_encodings_list = []
    for item in batch:
        positive_encodings_list.append(item['positive_encodings'])
    
    paths = [item['paths'] for item in batch]
  
    return {
        'anchor_input_ids': anchor_input_ids,
        'anchor_attention_mask': anchor_attention_mask,
        'positive_encodings_list': positive_encodings_list, # list of tensors
        'label': labels,
        'label_indices': label_indices,  # Keep label_indices as a list of tensors
        'paths': paths # list of list (each list contains a certain path)
    }

def get_top_level_ancestors(paths):
    return set(path[0] for path in paths if path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', default='data/BGC/bgc_train.json', type=str)
    parser.add_argument('--valid_dir', default='data/BGC/bgc_dev.json', type=str)
    parser.add_argument('--test_dir', default='data/BGC/bgc_test.json', type=str)
    parser.add_argument('--hierarchy_dir', default='data/BGC/bgc.taxonomy', type=str)
    parser.add_argument('--label_dir', default='data/BGC/labels.txt', type=str)
    parser.add_argument('--dataset', default='BGC', type=str)

    parser.add_argument('--backbone', default='bert-base-uncased', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--classification_loss', default='focal', type=str) # classification loss: choosing from BCE or focal
    
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--bert_lr', default=5e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lambda_contrastive', default=0.001, type=float)
    parser.add_argument('--lambda_classification', default=1, type=float)
    parser.add_argument('--lambda_hier', default=0.5,type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--gamma', default=2.0, type=float)
    parser.add_argument('--smoothing', default=0.2, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--min_delta', default=0.001, type=float)

    parser.add_argument('--save_model', default='model/model_weights.pth', type=str)


    args = parser.parse_args()

    # Load the dataset with descriptions and label list:
    f = open(args.train_dir, 'r')
    train_data = f.readlines()
    f.close()
    structured_train = []
    for item in train_data:
        item = json.loads(item)
        structured_train.append(item)

    f = open(args.valid_dir, 'r')
    dev_data = f.readlines()
    f.close()
    structured_dev = []
    for item in dev_data:
        item = json.loads(item)
        structured_dev.append(item)

    f = open(args.test_dir, 'r')
    test_data = f.readlines()
    f.close()
    structured_test = []
    for item in test_data:
        item = json.loads(item)
        structured_test.append(item)

    hierarchy = {}
    with open(args.hierarchy_dir, 'r') as file:
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

    with open(args.label_dir, 'r') as file:
        labels = [line.strip() for line in file]
    # Create label to index mapping
    label_to_index = {label: idx for idx, label in enumerate(sorted(labels))}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    hierarchy_index = {}
    # Convert the hierachy to use indices
    for child_label, parent_label in hierarchy.items():
        parent_index = label_to_index.get(parent_label)
        child_index = label_to_index.get(child_label)

        # if parent_index is not None and child_index is not None:
        if child_index is not None:
            hierarchy_index[child_index] = parent_index

    max_depth = compute_max_depth(hierarchy_index)

    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit(labels)

    # Initialize the Pre-trained LLM and Tokenizer
    model_name = args.backbone

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name)


    # Load and Preprocess Data:
    batch_size = args.bs

    train_dataset = MultipathDataset(structured_train[:5000], tokenizer, args.dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    eval_dataset = MultipathDataset(structured_dev[:1800], tokenizer, args.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    test_dataset = MultipathDataset(structured_test[:1800], tokenizer, args.dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    # Define the Model and Loss Function
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
   
    # Training Loop with Fine-Tuning
    classification_criterion = nn.BCEWithLogitsLoss()

    num_labels = len(labels)
    model = HierarchicalClassifier(args.backbone, num_labels)
    model = model.to(device)

    bert_parameters = list(model.text_encoder.parameters())
    projection_head_parameters = list(model.projection_head.parameters())
    classifier_parameters = list(model.classifier.parameters())
    log_temperature_parameter = [model.log_temperature]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.AdamW([
    #     {'params': bert_parameters, 'lr': args.bert_lr, 'weight_decay': args.weight_decay},
    #     {'params': projection_head_parameters, 'lr': args.lr, 'weight_decay': args.weight_decay},
    #     {'params': classifier_parameters, 'lr': args.lr, 'weight_decay': args.weight_decay},
    #     {'params': log_temperature_parameter, 'lr': args.lr, 'weight_decay': 0.0}  # No weight decay on temperature
    #     ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs / 2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warm-up

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
        )
    
    # Training loop
    focal_fn = HierarchicalFocalLossWithPenalty(
        hierarchy=hierarchy,
        gamma=args.gamma,
        lambda_hier=args.lambda_hier,
        smoothing=args.smoothing,
        reduction='mean'
    )

    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(args.num_epochs):
        print('start training!')
        model.train()
        total_loss = 0
        total_loss_classification = 0
        total_loss_contrastive = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()

            # Move data to device
            text_input_ids = batch['anchor_input_ids'].to(device)
            text_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_encodings_list = batch['positive_encodings_list']
            labels = batch['label'].to(device)
            anchor_labels_indices = batch['label_indices']  # List of labels per anchor (list of lists)
            # print("number of anchor: ", anchor_labels)
            paths = batch['paths']

            temperature = model.temperature

            # Forward pass to get anchor embeddings and logits
            logits, pooled_output = model(text_input_ids, text_attention_mask)  # [batch_size, num_labels], [batch_size, hidden_size]

            # Get projected embeddings for anchors
            anchor_embeddings = model.projection_head(pooled_output)   # [batch_size, projection_dim]

            # Compute classification loss
            if args.classification_loss == 'focal':
                # Compute focal loss 
                loss_classification = focal_fn(logits, labels)
            else:      
                # Compute hierarchical BCE loss
                loss_classification = hierarchical_bce_loss_with_penalty(
                    logits, labels, hierarchy, args.lambda_hier
                    )
            # print(f"Classification Loss before backward pass: {loss_classification.item()}")
            
            # Encode positive examples and associate them with paths
            positives_embeddings_list = []
            for i in range(anchor_embeddings.size()[0]):
                pos_items = positive_encodings_list[i]  # List of dicts with 'encodings' and 'path'
                if not pos_items:
                    positives_embeddings_list.append([])
                    continue

                embeddings_and_paths = []

                for pos_item in pos_items:
                    encoding = pos_item['encodings']
                    path = pos_item['path']
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    pos_embedding = model.encode(input_ids, attention_mask)  # [embedding_dim]
                    embeddings_and_paths.append({'embedding': pos_embedding, 'path': path})

                positives_embeddings_list.append(embeddings_and_paths)
        
            # Prepare negatives
            negatives_embeddings_list = []
            negatives_labels_list = []

            for i in range(anchor_embeddings.size()[0]):
                negatives_i = [] # Negatives for anchor i
                negative_labels_i = []
                anchor_paths = paths[i]  # Paths of the current anchor
                # anchor_top_ancestors = get_top_level_ancestors(paths[i])

                for j in range(anchor_embeddings.size()[0]):
                    if j == i:
                        continue  # Skip the anchor itself
                    
                    # # Check if any of the top-level ancestors match
                    # neg_paths = paths[j]
                    # neg_top_ancestors = get_top_level_ancestors(neg_paths)
                    # common_ancestors = anchor_top_ancestors.intersection(neg_top_ancestors)
                    
                    # if not common_ancestors:
                        # Add other anchor's embedding
                    negatives_i.append(anchor_embeddings[j].unsqueeze(0))
                    negative_labels_i.append(anchor_labels_indices[j])  # Labels of the negative (other anchor)

                    # Add positives of other anchors as negatives
                    for pos in positives_embeddings_list[j]:
                        pos_embedding = pos['embedding']
                        pos_path = pos['path']
                        negatives_i.append(pos_embedding)
                        negative_labels_i.append(pos_path)  # Use the path as labels for hierarchical similarity)

                        # Check if pos_path exactly matches any of the paths in anchor_paths
                        if pos_path not in anchor_paths:
                            negatives_i.append(pos_embedding)
                            negative_labels_i.append(pos_path)  # Use the path as labels for hierarchical similarity)

                if negatives_i:
                    negatives_i = torch.stack(negatives_i)  # [num_negatives_i, embedding_dim]
                else:
                    negatives_i = torch.empty(0, anchor_embeddings.size(1)).to(device)
                negatives_embeddings_list.append(negatives_i)
                negatives_labels_list.append(negative_labels_i)

            # Compute negative weights
            # negative_weights_list = compute_negative_weights(anchor_labels_indices, negatives_labels_list, hierarchy_index, anchor_embeddings.size()[0], device)

            # Compute contrastive loss
            # loss_contrastive = hierarchical_multi_positive_contrastive_loss(
            #     anchors=anchor_embeddings,
            #     positives_list=[ [pos['embedding'] for pos in pos_list] for pos_list in positives_embeddings_list ],
            #     negatives_list=negatives_embeddings_list,
            #     negative_weights_list=negative_weights_list,
            #     temperature=args.temperature
            #     )
            # loss_contrastive = hierarchical_info_nce_loss(anchors=anchor_embeddings, 
            #                                               positives_list=[ [pos['embedding'] for pos in pos_list] for pos_list in positives_embeddings_list ],
            #                                               negatives_list=negatives_embeddings_list, 
            #                                               negatives_labels_list=negatives_labels_list,
            #                                               labels_list=anchor_labels_indices, 
            #                                               hierarchy=hierarchy_index,
            #                                               temperature=temperature
            # )
            # print(f"Contrastive Loss before backward pass: {loss_contrastive.item()}")
            loss_contrastive = hierarchical_info_nce_loss(anchors = anchor_embeddings, 
                                                          positives_list=[ [pos['embedding'] for pos in pos_list] for pos_list in positives_embeddings_list ],
                                                          negatives_list=negatives_embeddings_list,
                                                          labels_list = anchor_labels_indices, 
                                                          positives_labels_list = [ [pos['path'] for pos in pos_list] for pos_list in positives_embeddings_list ], 
                                                          negatives_labels_list = negatives_labels_list, 
                                                          hierarchy = hierarchy_index, 
                                                          temperature = 0.07
                                                          )
            
            # Total loss
            total_loss_batch = args.lambda_contrastive * loss_contrastive + args.lambda_classification * loss_classification
            # print(f"Loss before backward pass: {total_loss_batch.item()}")
            # Backward pass and optimization
            total_loss_batch.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"No gradient for {name}")
            # Gradient clipping by norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_loss_batch.item()

            total_loss_classification += loss_classification.item()
            total_loss_contrastive += loss_contrastive.item()

            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # print(f"Total gradient norm: {total_norm}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")

        avg_classification_loss = total_loss_classification / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Classification Loss: {avg_classification_loss:.4f}")

        avg_contrastive_loss = total_loss_contrastive / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Contrastive Loss: {avg_contrastive_loss:.4f}")

        print('start evaluating!')
        model.eval()

        eval_all_labels = []
        eval_all_preds = []
        with torch.no_grad():
            for batch in eval_dataloader:
                text_input_ids = batch['anchor_input_ids'].to(device)
                text_attention_mask = batch['anchor_attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()

                # Forward pass
                outputs = model.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask
                    )
        
                text_embeddings = outputs.last_hidden_state[:, 0, :]
                logits = model.classifier(text_embeddings)
                preds = torch.sigmoid(logits).cpu().numpy()

                eval_all_labels.append(labels)
                eval_all_preds.append(preds)
    
        # Concatenate results
        all_labels = np.vstack(eval_all_labels)
        all_preds = np.vstack(eval_all_preds)

        # Binarize predictions with a threshold
        all_preds_bin = (all_preds >= args.threshold).astype(int)

        f1_micro = f1_score(all_labels, all_preds_bin, average='micro')

        # Binarize predictions with a threshold
        all_preds_bin = (all_preds >= args.threshold).astype(int)

        # Compute metrics
        f1_micro = f1_score(all_labels, all_preds_bin, average='micro')
        precision_micro = precision_score(all_labels, all_preds_bin, average='micro')
        recall_micro = recall_score(all_labels, all_preds_bin, average='micro')

        f1_macro = f1_score(all_labels, all_preds_bin, average='macro')
        precision_macro = precision_score(all_labels, all_preds_bin, average='macro')
        recall_macro = recall_score(all_labels, all_preds_bin, average='macro')

        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"Precision (Micro): {precision_micro:.4f}")
        print(f"Recall (Micro): {recall_micro:.4f}")

        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")

        # # Early Stopping
        # early_stopping(f1_micro)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    # Save model weights
    # torch.save(model.state_dict(), args.save_model)

    print('Finish Training!')
    # Inference
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            text_input_ids = batch['anchor_input_ids'].to(device)
            text_attention_mask = batch['anchor_attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            # Forward pass
            outputs = model.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embeddings = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(text_embeddings)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_preds.append(preds)

    # Concatenate results
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Binarize predictions with a threshold
    all_preds_bin = (all_preds >= args.threshold).astype(int)

    # Compute metrics
    f1_micro = f1_score(all_labels, all_preds_bin, average='micro')
    precision_micro = precision_score(all_labels, all_preds_bin, average='micro')
    recall_micro = recall_score(all_labels, all_preds_bin, average='micro')

    f1_macro = f1_score(all_labels, all_preds_bin, average='macro')
    precision_macro = precision_score(all_labels, all_preds_bin, average='macro')
    recall_macro = recall_score(all_labels, all_preds_bin, average='macro')

    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"Precision (Micro): {precision_micro:.4f}")
    print(f"Recall (Micro): {recall_micro:.4f}")

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")