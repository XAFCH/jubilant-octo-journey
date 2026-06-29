# PathSimNCE for Hierarchical Multi-Label Text Classification

## Overview

This repository contains code for hierarchical multi-label text classification with transformer-based text encoders, multi-label classification objectives, and hierarchy-aware contrastive learning over label paths. The main training and evaluation script for the current project is `running_multi_positive.py`.

The implemented training pipeline combines multi-label prediction with a path-level contrastive objective (`PathSimNCE`) and, when description-augmented data are provided, an additional description alignment objective. The repository also includes utilities for taxonomy processing, hierarchical path construction, evaluation, and visualization.

## Requirements

The implementation was developed with Python 3.10 and requires PyTorch, Transformers, NumPy, tqdm, scikit-learn, pandas, Matplotlib, and Seaborn. The OpenAI package is only required for optional LLM-based description generation.

## Data Preparation

Processed data files should be placed under:

- `data/BGC/`
- `data/NYT/`
- `data/WOS/`

The active dataset loader in `dataset.py` reads UTF-8 JSONL-style data. Each example include the following fields:

- `text`: input document text, depending on the processed dataset version
- `label`: ground-truth labels
- `path`: hierarchical label paths
- `path_description`: optional textual descriptions aligned with the paths

Please ensure that the processed files match the input text field expected by the dataset loader.

For description-aware experiments, use the processed files that include `path_description`, such as:

- `data/BGC/bgc_train_with_template_description.json`
- `data/BGC/bgc_val_with_template_description.json`
- `data/BGC/bgc_test_with_template_description.json`

Taxonomy and label files are stored alongside each dataset, for example:

- `data/BGC/bgc.taxonomy`
- `data/BGC/labels.txt`
- `data/NYT/nyt.taxonomy`
- `data/NYT/label.txt`
- `data/WOS/wos.taxonomy`
- `data/WOS/labels.txt`

Due to dataset license restrictions, raw datasets are not distributed with this anonymous repository. Please obtain the original data from their respective sources and place the processed files under the expected `data/` subdirectories.

## Training

The primary script for training and evaluation is `running_multi_positive.py`. To avoid environment-specific defaults, pass dataset and taxonomy paths explicitly.

For WOS and NYT, replace `--dataset_name`, `--train_path`, `--val_path`, `--test_path`, `--taxonomy_path`, and `--labels_path` with the corresponding files under `data/WOS/` or `data/NYT/`.

### WOS

```bash
python -u running_multi_positive.py \
  --dataset_name WOS \
  --train_path data/WOS/wos_train.json \
  --val_path data/WOS/wos_val.json \
  --test_path data/WOS/wos_test.json \
  --taxonomy_path data/WOS/wos.taxonomy \
  --labels_path data/WOS/labels.txt \
  --device cuda \
  --batch_size 32 \
  --max_length 256 \
  --epochs 50 \
  --patience 6 \
  --lr 3.5e-5 \
  --head_lr 1e-4 \
  --beta 0.1 \
  --beta_desc 0.1 \
  --tau 0.10 \
  --alpha 1.0 \
  --gamma 0.3 \
  --lam 0.3 \
  --hcl_candidate_mode sample \
  --hcl_space separate \
  --threshold_sweep \
  --threshold_sweep_steps 9 \
  --test_use_best_threshold \
  --hierarchy_decode \
  --encoder_name model/roberta-base/ \
  --desc_pos_agg avg \
  --multilabel_loss bce
```

### BGC

```bash
python -u running_multi_positive.py \
  --dataset_name BGC \
  --train_path data/BGC/bgc_train.json \
  --val_path data/BGC/bgc_val.json \
  --test_path data/BGC/bgc_test.json \
  --taxonomy_path data/BGC/bgc.taxonomy \
  --labels_path data/BGC/labels.txt \
  --device cuda \
  --batch_size 32 \
  --max_length 256 \
  --epochs 50 \
  --patience 6 \
  --lr 3.5e-5 \
  --head_lr 1e-4 \
  --beta 0.2 \
  --beta_desc 0.1 \
  --tau 0.07 \
  --alpha 2.0 \
  --gamma 0.3 \
  --lam 0.5 \
  --hcl_candidate_mode sample \
  --hcl_space separate \
  --threshold_sweep \
  --threshold_sweep_steps 9 \
  --test_use_best_threshold \
  --hierarchy_decode \
  --encoder_name model/roberta-base/ \
  --desc_pos_agg avg \
  --multilabel_loss bce
```

### NYT

```bash
python -u running_multi_positive.py \
  --dataset_name NYT \
  --train_path data/NYT/nyt_train.json \
  --val_path data/NYT/nyt_val.json \
  --test_path data/NYT/nyt_test.json \
  --taxonomy_path data/NYT/nyt.taxonomy \
  --labels_path data/NYT/label.txt \
  --device cuda \
  --batch_size 32 \
  --max_length 512 \
  --epochs 50 \
  --patience 6 \
  --lr 3.5e-5 \
  --head_lr 1e-4 \
  --beta 0.2 \
  --beta_desc 0.1 \
  --tau 0.07 \
  --alpha 2.0 \
  --gamma 0.3 \
  --lam 0.5 \
  --hcl_candidate_mode sample \
  --hcl_space separate \
  --threshold_sweep \
  --threshold_sweep_steps 9 \
  --test_use_best_threshold \
  --hierarchy_decode \
  --encoder_name model/roberta-base/ \
  --desc_pos_agg avg \
  --multilabel_loss bce
```

Commonly adjusted arguments include `--batch_size`, `--epochs`, `--patience`, `--lr`, `--head_lr`, `--tau`, `--alpha`, `--gamma`, `--lam`, `--threshold`, `--selection_metric`, and `--hierarchy_decode`.

## Evaluation

Threshold tuning is supported through `--threshold_sweep`. When `--test_use_best_threshold` is enabled, the best threshold selected on the validation set is reused for test evaluation.

During training, model checkpoints and threshold files are saved under `checkpoints/`. A saved checkpoint can also be evaluated directly using `--eval_only_checkpoint`. If a saved threshold file is available, it can be specified with `--eval_only_threshold_path`.

For example:

```bash
python running_multi_positive.py \
  --dataset_name BGC \
  --test_path data/BGC/bgc_test.json \
  --taxonomy_path data/BGC/bgc.taxonomy \
  --labels_path data/BGC/labels.txt \
  --encoder_name roberta-base \
  --eval_only_checkpoint checkpoints/EXAMPLE_best_model.pt \
  --eval_only_threshold_path checkpoints/EXAMPLE_best_threshold.txt
```

## Reproducing the Experiments

1. Prepare the processed dataset files under `data/BGC/`, `data/NYT/`, or `data/WOS/`.
2. Install the required Python packages listed in the Requirements section.
3. Run `running_multi_positive.py` with the desired training setting.
4. Enable `--threshold_sweep` to select the validation threshold.
5. Use `--test_use_best_threshold` during training-time evaluation, or `--eval_only_checkpoint` with an optional `--eval_only_threshold_path` for checkpoint-only testing.

## Project Structure

```text
.
|-- data/                        # Processed datasets, taxonomies, and label files
|-- outputs/                     # Saved figures and evaluation artifacts
|-- dataset.py                   # Dataset reader and collate function
|-- model.py                     # Transformer-based classification model
|-- hierarchical_infoNCE.py      # Hierarchy-aware contrastive losses
|-- losses_multilabel.py         # Multi-label loss functions
|-- hierarchy_from_taxonomy.py   # Taxonomy loading and hierarchy utilities
|-- eval.py                      # Evaluation helpers
|-- eval_multi_positive.py       # Evaluation helpers for multi-positive settings
|-- running_multi_positive.py    # Main training and evaluation script
`-- README.md
```

## Notes for Anonymous Review

- This repository is prepared for anonymous review.
- The README intentionally omits author-identifying information.
- Raw datasets are not redistributed in this repository.

## Citation

Citation information will be updated after the review process.
