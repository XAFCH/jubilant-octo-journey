import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import re
import pickle
import random
import time
from xml.dom import minidom
import xml.etree.ElementTree as ET
from tqdm import tqdm


train_data_path = '/Users/xiw4021/Downloads/blurbgenrecollectionen/BlurbGenreCollection_EN_train.txt'
dev_data_path = '/Users/xiw4021/Downloads/blurbgenrecollectionen/BlurbGenreCollection_EN_dev.txt'
test_data_path = '/Users/xiw4021/Downloads/blurbgenrecollectionen/BlurbGenreCollection_EN_test.txt'

# load the train_data as xml
data = ""
train_data = []
with open(train_data_path, 'r') as f:
    for line in f.readlines():
        data += line

train_data = ET.fromstring(data.replace('&', '&amp;'))
total_labels = 0
train_data_dict = []
for book in tqdm(train_data.findall('book'), total = len(train_data.findall('book'))):
    
    # check the keys
    title = book.find('title').text
    text = book.find('body').text
    i = 0
    labels = []
    topics = book.find('metadata').find('topics')

    while True:
        label = topics.findall(f'd{str(i)}')
        if len(label) == 0:
            break
        for l in label:
            total_labels += 1
            labels.append(l.text)
        i += 1
    train_data_dict.append({'token': 'Title: ' + title + '. ' + 'Text: ' + text, 'label': labels})

# store the train_data_dict as json lines
with open('./train_data.jsonl', 'w') as f:
    for data in train_data_dict:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')
total_labels

# do the same thing for dev_data
data = ""
dev_data = []
with open(dev_data_path, 'r') as f:
    for line in f.readlines():
        data += line

dev_data = ET.fromstring(data.replace('&', '&amp;'))

dev_data_dict = []
for book in dev_data.findall('book'):

    # check the keys
    title = book.find('title').text
    text = book.find('body').text
    i = 0
    labels = []
    topics = book.find('metadata').find('topics')

    while True:
        label = topics.findall(f'd{str(i)}')
        if len(label) == 0:
            break
        for l in label:
            labels.append(l.text)
        i += 1
    dev_data_dict.append({'token': 'Title: ' + title + '. ' + 'Text: ' + text, 'label': labels})
    
# store the dev_data_dict as json lines
with open('./dev_data.jsonl', 'w') as f:
    for data in dev_data_dict:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

# do the same thing for test_data
data = ""
test_data = []
with open(test_data_path, 'r') as f:
    for line in f.readlines():
        data += line

test_data = ET.fromstring(data.replace('&', '&amp;'))

test_data_dict = []
for book in test_data.findall('book'):
        
        # check the keys
        title = book.find('title').text
        text = book.find('body').text
        i = 0
        labels = []
        topics = book.find('metadata').find('topics')
    
        while True:
            label = topics.findall(f'd{str(i)}')
            if len(label) == 0:
                break
            for l in label:
                labels.append(l.text)
            i += 1
        test_data_dict.append({'token': 'Title: ' + title + '. ' + 'Text: ' + text, 'label': labels})

# store the test_data_dict as json lines
with open('./test_data.jsonl', 'w') as f:
    for data in test_data_dict:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


# load the hierarchy
hierarchy = {}
with open('/Users/xiw4021/Documents/HTMC/data/BGC/bgc.taxonomy', 'r') as file:
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

def get_full_label_list(label, hierarchy):
    """Traverse the hierarchy from a label up to the root and return the full path."""
    full_label_list = []
    current_label = label

    # Traverse the hierarchy upwards to the root
    while current_label is not None:
        full_label_list.append(current_label)
        current_label = hierarchy.get(current_label, None)  # Move to the parent

    return full_label_list[::-1]  # Reverse the list to have root-to-leaf order


def remove_redundant_paths(paths):
    """Remove paths that are sub-paths of others by comparing the elements."""
    # Sort paths by length (longest first) so longer paths are checked first
    paths = sorted(paths, key=len, reverse=True)

    unique_paths = []

    for current_path in paths:
        is_subpath = False
        for existing_path in unique_paths:
            # Check if current_path is a sub-path of any longer path in unique_paths
            if len(current_path) <= len(existing_path) and current_path == existing_path[:len(current_path)]:
                is_subpath = True
                break
        if not is_subpath:
            unique_paths.append(current_path)

    # Remove 'Root' from each path
    unique_paths = [path[1:] if path[0] == 'Root' else path for path in unique_paths]

    return unique_paths

new_train_data = []
for data in tqdm(train_data):
    all_path = [get_full_label_list(label, hierarchy) for label in data['label']]
    unique_paths = remove_redundant_paths(all_path)
    data['path'] = unique_paths
    new_train_data.append(data)
with open('data/BGC/bgc_train_raw.json', "w") as file:
    for entry in new_train_data:
        file.write(json.dumps(entry) + "\n")

new_dev_data = []
for data in tqdm(dev_data):
    all_path = [get_full_label_list(label, hierarchy) for label in data['label']]
    unique_paths = remove_redundant_paths(all_path)
    data['path'] = unique_paths
    new_dev_data.append(data)
with open('data/BGC/bgc_dev_raw.json', "w") as file:
    for entry in new_dev_data:
        file.write(json.dumps(entry) + "\n")

new_test_data = []
for data in tqdm(test_data):
    all_path = [get_full_label_list(label, hierarchy) for label in data['label']]
    unique_paths = remove_redundant_paths(all_path)
    data['path'] = unique_paths
    new_test_data.append(data)
with open('data/BGC/bgc_test_raw.json', "w") as file:
    for entry in new_test_data:
        file.write(json.dumps(entry) + "\n")
