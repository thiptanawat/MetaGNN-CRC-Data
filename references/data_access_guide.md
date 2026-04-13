# Data & Reference Access Guide — MetaGNN-CRC Dataset (Data in Brief)

**Author:** Thiptanawat Phongwattana
**Affiliation:** School of Information Technology, KMUTT
**Corresponding:** Jonathan H. Chan

---

## Primary Dataset Access

The MetaGNN-CRC dataset described in this Data in Brief article will be
deposited on Zenodo upon manuscript acceptance. The provisional DOI
placeholder in the manuscript (`10.5281/zenodo.XXXXXXX`) will be updated
at that time.

**Dataset Components** (see `raw_data/dataset_manifest.csv` for full inventory):

| Component | Source | Size | Restricted? |
|-----------|--------|------|-------------|
| RNA-seq VST matrix (60,660 genes × 219 patients) | TCGA GDC | ~230 MB | No (dbGaP open-tier) |
| Proteomics TMT matrix (11,348 proteins × 219 patients) | CPTAC PDC | ~95 MB | No (public) |
| Recon3D-mapped graph tensors | Derived | ~45 MB | No |
| Pre-trained MetaGNN weights | Derived | ~8 MB | No |
| Clinical metadata (TSV) | TCGA | ~45 KB | No |
| Recon3D full model (.mat) | VMH Life | ~320 MB | No (public) |
| HMA tissue-specific GEMs | Metabolic Atlas | ~410 MB | No (public) |

---

## Reference Access Table

### Dataset Source Papers

| # | Citation | DOI | Access |
|---|----------|-----|--------|
| 1 | TCGA Colorectal Cancer (Nature 2012) | https://doi.org/10.1038/nature11252 | **Open Access** |
| 2 | Vasaikar et al. 2019 — CPTAC-CRC proteomics (Cell) | https://doi.org/10.1016/j.cell.2019.07.012 | **Open Access** |
| 3 | Brunk et al. 2018 — Recon3D (Nat Chem Biol) | https://doi.org/10.1038/nchembio.2304 | Paywalled — institutional library |
| 4 | Human Metabolic Atlas (Sci Signal 2021) | https://doi.org/10.1126/scisignal.abj1541 | **Open Access** |
| 5 | Wilkinson et al. 2016 — FAIR principles (Sci Data) | https://doi.org/10.1038/sdata.2016.18 | **Open Access** |
| 6 | Hoadley et al. 2018 — Pan-Cancer cell-of-origin (Cell) | https://doi.org/10.1016/j.cell.2018.03.05 | **Open Access** |

### Methodology Papers

| # | Citation | DOI | Access |
|---|----------|-----|--------|
| 7 | Love et al. 2014 — DESeq2 VST (Genome Biol) | https://doi.org/10.1186/s13059-014-0550-8 | **Open Access** |
| 8 | Ritchie et al. 2015 — limma (Nucleic Acids Res) | https://doi.org/10.1093/nar/gkv007 | **Open Access** |
| 9 | Zur et al. 2010 — GPR convention (Bioinformatics) | https://doi.org/10.1093/bioinformatics/btq602 | Paywalled — institutional library |
| 10 | Morgan (2010) — Morgan fingerprints | N/A (algorithm described in RDKit) | N/A — use RDKit |

---

## Step-by-Step Data Download Instructions

### A. TCGA RNA-seq Data (GDC Portal)

1. Navigate to https://portal.gdc.cancer.gov/
2. Create a free account (open-access data does not require dbGaP approval)
3. Filter: Project = TCGA-COAD OR TCGA-READ
4. Filter: Data Category = Transcriptome Profiling → Gene Expression Quantification → STAR - Counts
5. Add to cart and download the GDC manifest file
6. Install and run `gdc-client`:

```bash
# Download gdc-client (Linux)
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip && chmod +x gdc-client

# Download data using manifest
./gdc-client download -m gdc_manifest.txt -d ./tcga_star_counts/ -n 8
```

Expected: ~219 STAR count TSV files for COAD (155) + READ (64) primary tumour samples.

---

### B. CPTAC Proteomics Data (PDC Portal)

1. Navigate to https://pdc.cancer.gov/
2. Register for a free account
3. Search Study: **PDC000116** (CPTAC Colon Cancer) and **PDC000220** (CPTAC Rectal Cancer)
4. Download: **Protein Assembly** → TMT ratio matrix (log2 normalised)
5. Download: **Clinical manifest** (maps aliquot IDs → case IDs → TCGA barcodes)

Alternatively, use the PDC API:
```bash
# PDC GraphQL API — query protein abundance for a study
curl -X POST https://pdc.cancer.gov/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ proteinAbundance(study_id: \"PDC000116\", resolution: \"gene\") { gene_name aliquot_id log2_ratio } }"}'
```

---

### C. Human Metabolic Atlas (HMA)

```bash
# Free download, no account required
wget -O Human1_GEMs.zip \
  https://metabolicatlas.org/downloads/Human1/Human1_GEMs_v2.0.zip
unzip Human1_GEMs.zip -d ./hma_gems/
# Contains 98 tissue-specific GEMs in JSON + MATLAB formats
```

---

### D. Recon3D Genome-Scale Metabolic Model

```bash
# Option 1: VMH Life database (recommended — direct download)
# Visit: https://www.vmh.life/#downloadview
# Select: Recon 3D → Download COBRA MATLAB format (.mat)

# Option 2: BiGG Database
# Visit: http://bigg.ucsd.edu/models/Recon3D
# Download Recon3D.mat (COBRA format) or Recon3D.xml (SBML)
```

---

### E. PubChem Physico-Chemical Properties for Metabolites

Used to construct the metabolite node feature matrix X_M (519-dim).
The property TSV can be regenerated using the PubChem REST API:

```python
# fetch_pubchem_props.py
import requests, pandas as pd, time

bigg_to_inchikey = {}  # Load from Recon3D metabolite file

records = []
for bigg_id, inchikey in bigg_to_inchikey.items():
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"inchikey/{inchikey}/property/"
           f"MolecularWeight,XLogP,HBondAcceptorCount,HBondDonorCount,"
           f"TPSA,RingCount,FormalCharge,IsomericSMILES/JSON")
    try:
        resp = requests.get(url, timeout=10).json()
        props = resp['PropertyTable']['Properties'][0]
        props['bigg_id'] = bigg_id
        records.append(props)
    except Exception:
        pass
    time.sleep(0.2)  # Respect PubChem rate limit (5 req/sec)

pd.DataFrame(records).to_csv('pubchem_metabolite_props.tsv', sep='\t', index=False)
```

---

## FAIR Compliance Notes

This dataset adheres to FAIR data principles (Wilkinson et al. 2016):

- **Findable:** Zenodo DOI + metadata registered with DataCite
- **Accessible:** All files publicly downloadable via Zenodo (no authentication required for derived tensors)
- **Interoperable:** HDF5 (standard), TSV (plain text), PyTorch .pt (documented format), MATLAB .mat (COBRA convention)
- **Reusable:** MIT licence; provenance documented in `clinical_metadata.tsv` TCGA barcode column; preprocessing code provided in `code/`

---

*Contact: thiptanawat.phon@sit.kmutt.ac.th | jonathan@sit.kmutt.ac.th*
