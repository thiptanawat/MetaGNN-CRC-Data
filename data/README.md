# MetaGNN-CRC Data Directory

This directory contains the dataset files for MetaGNN-CRC. Files are organized into three subdirectories: raw, processed, and checkpoints.

## Directory Structure

```
data/
├── raw/                          # Downloaded source files (not in git)
│   ├── TCGA_COAD_READ_RNA.h5     # TCGA RNA-seq expression matrix
│   ├── Recon3D_v3.xml             # Metabolic network model (SBML)
│   ├── DepMap_22Q4.csv            # Dependency scores
│   ├── HMA_tissue_gems/           # Human Metabolic Atlas tissue models
│   └── checksums.json             # SHA-256 hashes for integrity verification
│
├── processed/                     # Pre-computed tensors (PyTorch format)
│   ├── patient_features.pt        # Patient node features (v1)
│   ├── patient_features_v2.pt     # Patient node features (v2, enriched)
│   ├── graph_structure.pt         # Heterogeneous graph structure
│   ├── hma_labels_thresholded.pt  # Reaction activity labels
│   ├── gpr_reaction_mask.pt       # Mask for GPR-mapped reactions
│   ├── evaluation_subset_ids.json # 220-patient benchmark indices
│   └── cv_splits_624/             # 5-fold CV indices
│       ├── fold_1.json
│       ├── fold_2.json
│       ├── fold_3.json
│       ├── fold_4.json
│       └── fold_5.json
│
└── checkpoints/                   # Trained model weights
    ├── config_220.json            # Hyperparameters (220-patient)
    ├── config_624.json            # Hyperparameters (624-patient)
    ├── metagnn_220_seed2024.pt    # Fine-tuned weights
    ├── metagnn_220_seed42.pt
    ├── metagnn_220_seed123.pt
    ├── metagnn_624_fold1.pt       # CV fold weights
    ├── metagnn_624_fold2.pt
    ├── metagnn_624_fold3.pt
    ├── metagnn_624_fold4.pt
    └── metagnn_624_fold5.pt
```

## File Descriptions

### Raw Data (`raw/`)

Files in this directory are downloaded from external sources using `scripts/download_raw_data.py`.

- **TCGA_COAD_READ_RNA.h5**: TCGA colorectal cancer RNA-seq data
  - Format: HDF5 (expression matrix)
  - Size: ~178 MB
  - Patients: 690 (COAD + READ cohorts)
  - Genes: 40,799
  - Source: GDC Portal (https://portal.gdc.cancer.gov)

- **Recon3D_v3.xml**: Human metabolic reference network
  - Format: SBML (Systems Biology Markup Language)
  - Size: ~8 MB
  - Reactions: 10,600
  - Metabolites: 5,835
  - Source: vmh.life

- **DepMap_22Q4.csv**: Cancer dependency scores
  - Format: CSV
  - Size: ~450 MB
  - Provides genome-scale pathway context
  - Source: DepMap Public Portal

- **HMA_tissue_gems/**: Tissue-specific metabolic models
  - 11 tissue types (colon, liver, brain, etc.)
  - Format: MATLAB .mat files
  - Source: Human Metabolic Atlas GitHub

- **checksums.json**: SHA-256 hashes
  - Verify file integrity after download
  - Format: JSON key-value pairs

### Processed Data (`processed/`)

Pre-computed tensors and graph structures ready for model training. These are available as a Zenodo archive.

**Tensor Files (PyTorch .pt format):**

- **patient_features.pt**: v1 patient features
  - Shape: (690, 2)
  - Channels: [GPR-mapped expression, zero-filled proteomics]
  - Dtype: float32

- **patient_features_v2.pt**: v2 patient features (enriched)
  - Shape: (690, 3)
  - Channels: [mean expression, max expression, fraction above median]
  - Dtype: float32
  - Recommended for training

- **graph_structure.pt**: Heterogeneous graph representation
  - Node types: reaction, metabolite, gene
  - Edge types: catalyzes, substrate_of, produces, shared_metabolite
  - Format: PyTorch Geometric Data object
  - Dtype: int64 (edges), float32 (features)

- **hma_labels_thresholded.pt**: Reaction activity labels
  - Shape: (10600,)
  - Values: binary {0, 1}
  - Active: 7,434 reactions (70.1%)
  - Inactive: 3,166 reactions (29.9%)
  - Derived from expression thresholding + tissue consensus

- **gpr_reaction_mask.pt**: GPR-mapped reaction indicator
  - Shape: (10600,)
  - Values: binary {0, 1}
  - 1: reaction has Gene-Protein-Reaction (GPR) rule (5,937 reactions)
  - 0: reaction lacks GPR rule (4,663 reactions)
  - Use to subset analyses to high-confidence reactions

**Index Files (JSON):**

- **evaluation_subset_ids.json**: 220-patient benchmark subset
  - Contains: list of patient indices
  - Train/test split: 80/20 (176 train, 44 test)
  - Used in published benchmarks

- **cv_splits_624/**: 5-fold stratified cross-validation indices
  - Each fold contains: train_indices, test_indices
  - Stratification: reaction activity distribution
  - Used for full cohort validation

### Checkpoints (`checkpoints/`)

Fine-tuned model weights and hyperparameter configurations.

**Configuration Files:**

- **config_220.json**: Training hyperparameters for 220-patient benchmark
  - Learning rate, batch size, epoch count, etc.
  - Achieves: F1 0.796, AUROC 0.861

- **config_624.json**: Training hyperparameters for 624-patient full cohort
  - Cross-validation specific settings
  - Achieves: F1 0.445, AUROC 0.663

**Checkpoint Files:**

- **metagnn_220_seed*.pt**: Trained models (220-patient runs)
  - Seeds: 2024, 42, 123
  - Parameters: 143,489
  - Size: ~576 KB each

- **metagnn_624_fold*.pt**: Trained models (5-fold CV)
  - Folds: 1, 2, 3, 4, 5
  - Each trained on 4 folds, evaluated on 1 fold

## Expected File Counts and Sizes

| Category | Count | Total Size |
|----------|-------|-----------|
| Raw data files | 4 | ~640 MB |
| Processed tensors | 5 | ~200 MB |
| Index files | 6 | ~2 MB |
| Checkpoints | 8 | ~5 MB |
| **Total (archive)** | **23** | **~850 MB** |

## Zenodo Archive

Pre-computed `processed/` and `checkpoints/` directories are available on Zenodo for fast setup:

**DOI:** https://doi.org/10.5281/zenodo.18903519

Download and extract to `data/` to skip preprocessing steps.

Raw data files must be downloaded separately using `scripts/download_raw_data.py`.

## File Naming Conventions

- **Tensors**: `{feature}_{description}.pt` (e.g., `patient_features_v2.pt`)
- **Indices**: `{subset}_{type}.json` (e.g., `cv_splits_624.json`)
- **Checkpoints**: `metagnn_{cohort}_seed{seed}.pt` or `metagnn_{cohort}_fold{fold}.pt`
- **Configs**: `config_{cohort}.json`

## Loading Data in Python

### Load processed tensors
```python
import torch

# Load patient features
features = torch.load('data/processed/patient_features_v2.pt', weights_only=True)
print(f"Shape: {features.shape}")  # (690, 3)

# Load reaction labels
labels = torch.load('data/processed/hma_labels_thresholded.pt', weights_only=True)
print(f"Active: {labels.sum().item()}, Inactive: {(labels == 0).sum().item()}")

# Load graph
graph = torch.load('data/processed/graph_structure.pt', weights_only=False)
print(f"Nodes: {graph.x.shape}, Edges: {graph.edge_index.shape}")
```

### Load checkpoint
```python
checkpoint = torch.load('data/checkpoints/metagnn_220_seed2024.pt')
# checkpoint is a dict with keys: 'model_state_dict', 'optimizer_state_dict', 'epoch', etc.
```

### Load cross-validation splits
```python
import json

with open('data/processed/cv_splits_624/fold_1.json') as f:
    fold_data = json.load(f)
    train_idx = fold_data['train_indices']
    test_idx = fold_data['test_indices']
```

## Data Integrity

Verify all files after download or extraction:

```bash
python scripts/verify_data.py --data_dir data/
```

This checks tensor shapes, graph structure, and file integrity.

## License and Attribution

- **Dataset**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Code**: MIT License

See repository LICENSE file for full details.

---

**Last updated:** 2025
