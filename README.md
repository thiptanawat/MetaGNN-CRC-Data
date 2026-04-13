# MetaGNN-CRC Dataset

**Manuscript:** "MetaGNN-CRC: A Recon3D-Mapped Transcriptomic Dataset with Proteomics-Ready Architecture for Graph-Based Metabolic Network Reconstruction in Colorectal Cancer"

**Journal:** Data in Brief (Elsevier)
**Authors:** Thiptanawat Phongwattana, Jonathan H. Chan*
**Affiliation:** School of Information Technology, King Mongkut's University of Technology Thonburi (KMUTT), 126 Pracha Uthit Rd., Bang Mod, Thung Khru, Bangkok 10140, Thailand
*Corresponding author: jonathan@sit.kmutt.ac.th

**Data Archive (Zenodo):** [https://doi.org/10.5281/zenodo.18903519](https://doi.org/10.5281/zenodo.18903519)

---

## Repository Structure

This GitHub repository contains the source code, raw summary data, and pre-computed results. The full processed dataset (graph tensors, patient features, model checkpoints) is archived on Zenodo due to file size constraints.

```
MetaGNN-CRC/
├── README.md                                    # This file
├── MANIFEST.sha256                              # SHA-256 checksums for all data files
├── raw_data/                                    # CSV data underlying each figure
│   ├── dib_fig1_omics_coverage.csv              # Fig 1A: patient omics coverage
│   ├── dib_fig1_stage_distribution.csv          # Fig 1B: AJCC tumour stage
│   ├── dib_fig1_msi_status.csv                  # Fig 1C: MSI classification
│   ├── dib_fig2_rnaseq_vst_distribution.csv     # Fig 2A: VST expression distribution
│   ├── dib_fig2_protein_completeness.csv        # Fig 2B: proteomics completeness curve
│   ├── dib_fig2_recon3d_reaction_coverage.csv   # Fig 2C: reaction coverage by omics
│   └── dataset_manifest.csv                     # Full inventory of all dataset files
├── code/
│   ├── requirements.txt                         # Python dependencies
│   ├── 00_download_and_generate_690.py          # Download and generate full 690-patient dataset
│   ├── 01_preprocess_tcga_rnaseq.py             # RNA-seq VST normalisation pipeline
│   ├── 02_preprocess_cptac_proteomics.py        # TMT proteomics normalisation pipeline
│   ├── 03_construct_hetero_graph.py             # Graph tensor construction
│   ├── 04_generate_dib_figures.py               # All DIB manuscript figures
│   ├── 05_validate_dataset.py                   # Dataset integrity checker
│   ├── download_all_raw_data.py                 # Raw data download utilities
│   ├── download_cptac_correct.py                # CPTAC data retrieval
│   └── download_cptac_pdc.py                    # PDC data retrieval
├── notebooks/                                   # Jupyter notebooks for exploration
│   ├── 01_Cohort_Overview.ipynb
│   ├── 02_Data_Quality_Assessment.ipynb
│   ├── 02_Data_Quality_Assessment_v2.ipynb
│   └── 03_Dataset_Validation.ipynb
├── results/                                     # Pre-computed results
│   ├── cohort_demographics/                     # Patient metadata & age distribution
│   ├── dataset_validation/                      # Integrity reports & file manifests
│   ├── graph_statistics/                        # Edge/node statistics & degree distribution
│   ├── hma_matching/                            # HMA tissue-model label matching
│   └── omics_qc/                                # RNA-seq & proteomics quality control
└── references/
    └── data_access_guide.md                     # Download instructions for all data sources
```

---

## Dataset Overview

MetaGNN-CRC is a curated, Recon3D-mapped transcriptomic dataset with proteomics-ready architecture for graph-based metabolic network reconstruction in colorectal cancer. It provides RNA-seq data for 690 TCGA colorectal adenocarcinoma patients pre-processed and mapped to the Recon3D v3 metabolic network as heterogeneous graph-structured tensors, directly loadable into PyTorch Geometric.

**Value:** No existing public dataset bridges the gap between genome-scale metabolic models (GEMs) and graph neural networks. Preparing this data from scratch requires 2–4 weeks of specialised bioinformatics work. MetaGNN-CRC eliminates this bottleneck.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total patients | 690 TCGA-CRC (COAD + READ cohorts) |
| Transcriptomics | RNA-seq TPM (40,799 genes × 690 patients, 178 MB) |
| Proteomics | Zero-filled for all 690 patients (architecture-ready; see Limitations) |
| Reference model | Recon3D v3 |
| Reaction nodes | 10,600 (5,937 with GPR rules; 4,663 without) |
| Metabolite nodes | 5,835 |
| Genes (GPR-mapped) | 2,248 |
| Stoichiometric edges | 40,425 (20,512 substrate_of + 19,913 produces) |
| Shared_metabolite edges | 7,517,742 undirected (~83,306 at default k=10 sparsification) |
| Metabolite features | 519-dimensional (7 physico-chemical + 512-bit Morgan fingerprints) |
| Expression-thresholded labels | 7,434 active / 3,166 inactive (70.1% / 29.9%) |
| Model checkpoint | 143,489 parameters, 576 KB |
| Licence | CC-BY 4.0 |

### Label Files (on Zenodo)

Three label sets are associated with this dataset. **Use `hma_labels_thresholded.pt` for training.**

| File / Source | Active / Inactive | Notes |
|---------------|-------------------|-------|
| `hma_labels.pt` | 10,600 / 0 | Generic HMA bounds (all active); NOT recommended for training |
| `hma_labels_thresholded.pt` | 7,434 / 3,166 | Expression-thresholded consensus (RECOMMENDED); "hma" prefix is historical artefact |
| 11-tissue union (code-generated) | 8,147 / 2,453 | Generated dynamically by code; not deposited as a file |

### Feature Versions (on Zenodo)

| Version | Dimensions | Description |
|---------|-----------|-------------|
| v1 (`patient_features/`) | 10,600 × 2 | Scalar GPR-mapped expression + zero-filled proteomic channel |
| v2 (`patient_features_v2/`) | 10,600 × 3 | Enriched: mean expression, max expression, fraction above cohort median |

The trained model checkpoint uses v2 features.

---

## Quick Start

### Reproduce Figures from Raw Data

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Reproduce DIB figures from CSV raw data
python code/04_generate_dib_figures.py

# 3. Validate dataset integrity (after downloading full dataset from Zenodo)
python code/05_validate_dataset.py ./path/to/zenodo/download/
```

### Full Preprocessing Pipeline

To regenerate the complete dataset from source, download the raw TCGA/CPTAC files first (see `references/data_access_guide.md`), then run:

```bash
python code/01_preprocess_tcga_rnaseq.py \
    --gdc_dir ./tcga_star_counts/ \
    --manifest gdc_manifest.txt \
    --recon3d_genes recon3d_gene_list.txt \
    --output_dir ./processed/rnaseq/

python code/02_preprocess_cptac_proteomics.py \
    --pdc_tsv cptac_crc_tmt_abundance.tsv \
    --pdc_clinical cptac_clinical.tsv \
    --tcga_barcodes tcga_barcodes.txt \
    --recon3d_genes recon3d_gene_list.txt \
    --output_dir ./processed/proteomics/

python code/03_construct_hetero_graph.py \
    --recon3d_mat Recon3D.mat \
    --rnaseq_h5 ./processed/rnaseq/tcga_crc_rnaseq_vst.h5 \
    --proteomics_h5 ./processed/proteomics/cptac_crc_protein_tmt.h5 \
    --pubchem_props_tsv pubchem_metabolite_props.tsv \
    --gpr_table_tsv recon3d_gpr_table.tsv \
    --hma_mat Human1_GEMs.mat \
    --output_dir ./graph_data/
```

### Load Processed Data (from Zenodo)

```python
import torch

# Load graph structure
graph = torch.load('data/processed/graph_structure.pt')

# Load patient features
patient = torch.load('data/processed/patient_features_v2/TCGA-A6-2671.pt',
                     weights_only=True)

# Load labels
labels = torch.load('data/processed/hma_labels_thresholded.pt',
                    weights_only=True)

# Verify shapes
assert graph['reaction'].num_nodes == 10600
assert graph['metabolite'].num_nodes == 5835
assert patient.shape == (10600, 3)
assert labels.shape[0] == 10600
assert labels.sum().item() == 7434

print("All checks passed. Dataset is ready for use.")
```

---

## Limitations

1. **Pseudo-labels, not ground truth.** Expression-thresholded labels are surrogate labels derived from gene expression via GPR rules and cohort majority vote, not experimentally measured reaction fluxes. Users with higher-quality supervision (e.g., ¹³C flux measurements, DepMap essentiality) should replace them.

2. **Proteomic channel zero-filled.** Two CPTAC studies exist: PDC000111 (TCGA Retrospective, ~90 patients, Label Free) provides only raw spectra without a pre-computed quantitative matrix; PDC000116 (CPTAC Prospective, 102 patients, TMT) is an independent cohort with zero TCGA patient overlap. The proteomic feature channel is therefore zero-filled for all 690 patients. A proteomics ablation confirms negligible impact (ΔF1 = +0.0016).

3. **Non-GPR reaction bias.** 4,663 of 10,600 reactions (44%) lack GPR rules and receive a default "active" label, inflating the active class. Use `gpr_reaction_mask.pt` to restrict analyses to the 5,937 GPR-mapped reactions.

4. **Subsystem coverage.** 3,961 reactions (37.4%) could not be mapped to metabolic subsystems via the Human-GEM BiGG cross-reference. These are predominantly transport and exchange reactions. They participate in training but are excluded from subsystem-level analysis.

5. **Gene symbol collisions.** HGNC symbols with duplicate Ensembl mappings are resolved by retaining the first occurrence. Users requiring Ensembl-level resolution should remap from raw GDC files.

---

## Recommended Evaluation Protocol

1. Use the 220-patient subset (`evaluation_subset_ids.json`) with expression-thresholded labels for benchmarking (AUROC 0.861, F1 0.796).
2. Report both all-reaction (10,600) and GPR-only (5,937) metrics via `gpr_reaction_mask.pt`.
3. Use AUROC as the primary threshold-independent metric, supplemented by F1, precision, and recall.
4. Recompute consensus pseudo-labels from training-split patients only to avoid transductive label leakage.
5. Specify the shared_metabolite sparsification parameter k for reproducibility.

---

## Reproducibility

- All data files include SHA-256 checksums in `MANIFEST.sha256`
- Validation script: `code/05_validate_dataset.py`
- All dependency versions pinned in `code/requirements.txt`
- Archived on Zenodo with DOI for long-term persistence

---

## Ethics

All data are de-identified and accessed via standard TCGA data use agreements (dbGaP phs000178). No new patient samples were collected. No additional institutional ethical approval was required.

---

## Licence

- **Dataset:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Code:** MIT Licence
