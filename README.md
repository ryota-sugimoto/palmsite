# **PalmSite — RdRP catalytic center predictor**

PalmSite is a fast command-line tool that predicts the **RNA-dependent RNA polymerase (RdRP) catalytic center** from protein FASTA and outputs **GFF3**.
As of **v0.1.2**, PalmSite can also optionally output **per-residue attention weights** and **span parameters** in **JSON**.

---

## **Highlights**

* **One command** from FASTA → GFF3:

  ```bash
  palmsite <fasta ...>
  ```
* **New:** optional **JSON output** of residue-wise attention and span details:

  ```bash
  palmsite --attn-json details.json <fasta>
  ```
* **High precision and recall AUC** (internal benchmarks):

| Backbone (ESM-C) | Positives vs. Negatives | Positives vs. Rest |
| ---------------- | ----------------------- | ------------------ |
| **6b**           | 0.9998                  | 0.9848             |
| **600m**         | 0.9992                  | 0.9687             |
| **300m**         | 0.9991                  | 0.9755             |

* Detects **distant homologs** (e.g., HSRV RdRP in Urayama et al., 2024).

---

## **Installation**

```bash
conda create -n palmsite python=3.11
conda activate palmsite
pip install palmsite
```

---

## **Quickstart**

```bash
# Basic (default backbone: 600m, local)
palmsite -o hsrv_rdrp-domain.gff examples/hsrv_proteins.fasta

# Or write to stdout
palmsite examples/hsrv_proteins.fasta > hsrv_rdrp-domain.gff

# Quiet mode
palmsite -q examples/sars-cov-2_proteins.fasta

# Increase reporting threshold
palmsite -p 0.9 examples/zikavirus_proteins.fasta

# Use 6B (Forge)
palmsite -b 6b -k <FORGE_TOKEN> examples/turnip-mosaic-virus_proteins.fasta
```

Notes:

* `-b/--backbone` selects the ESM-C embedding model: **300m**, **600m** (local), or **6b** (Forge).
* For `6b`, set `-k <token>` or export `ESM_FORGE_TOKEN`.

---

## **NEW: Attention JSON output**

PalmSite now supports optional **per-residue attention-weight output** in JSON format:

```bash
palmsite \
  -o result.gff \
  --attn-json attention_details.json \
  examples/myproteins.fasta
```

Each entry corresponds to one **embedded chunk** and includes:

```json
{
  "chunk_id": {
    "L": <length>,
    "orig_start": <absolute_start>,
    "orig_len": <protein_length>,
    "mu": <anchor_mu>,
    "sigma": <anchor_sigma>,
    "mu_attn": <gaussian_mu>,
    "sigma_attn": <gaussian_sigma>,
    "S_norm": <span_start_norm>,
    "E_norm": <span_end_norm>,
    "S_idx": <span_start_index>,
    "E_idx": <span_end_index>,
    "P": <probability>,
    "w": [... per-residue attention weights ...],
    "abs_pos": [... absolute positions ...]
  }
}
```

---

## **Command-line usage**

```
Usage: palmsite [OPTIONS] [FASTAS]...

PalmSite — RdRP catalytic center predictor.
Usage: palmsite -p 0.5 [-o result.gff] [--attn-json details.json] <fasta ...>
```

### **Options**

```
  --version                       Show version and exit
  -o, --gff-out PATH              Write GFF3; default: stdout
  -p, --min-p FLOAT               Minimum probability for GFF [default: 0.5]
  -b, --backbone [300m|600m|6b]   Embedding backbone (local or Forge)
  -m, --model-id TEXT             HF model repo (default: ryota-sugimoto/palmsite)
  -d, --device [auto|cpu|cuda]    Device for local models
  -k, --token TEXT                Forge token for 6B
  -t, --tmp-dir PATH              Temp directory
  -q, --quiet                     Suppress logs
  -v, --verbose                   Debug logs (overrides quiet)
  --keep-tmp                      Keep temp files
  --attn-json PATH                Write per-residue attention JSON
  FASTAS...                       One or more FASTA files
```

---

## **What PalmSite does**

### 1. **Sanitize & merge FASTA**

Removes unusual characters, replaces with `X`, drops sequences with too many corrections, and writes a clean merged FASTA.
(src: `sanitize.py`) 

### 2. **Embed sequences**

The embedding engine (`_embed_impl.py`) generates an **HDF5** file containing token-wise ESM-C embeddings:

* **300m / 600m** — local Hugging Face models
* **6B** — via **ESM Forge** API

The CLI invokes this via `embed_shim.py`. 
Embedding logic: `_embed_impl.py` 

### 3. **Predict RdRP domains**

Prediction code lives in:

* `_predict_impl.py` (full engine with CSV, GFF3, HDF5 export, and JSON export) 
* `infer_simple.py` (minimal GFF3 generator, now with JSON support) 

Outputs include:

* **GFF3** spans
* **(New) JSON** with attention maps

---

## **Output files**

### **1. GFF3 (default)**

Contains one feature per protein:

| Attribute                 | Meaning                    |
| ------------------------- | -------------------------- |
| `P`                       | RdRP probability           |
| `sigma`                   | attention span width       |
| `Chunk` / `ChunkOrWindow` | source chunk or window     |
| `SpanSource`              | `kSigma` or `HPD`          |
| `AttnMass`                | HPD mass used (if enabled) |
| `AttnEntropy`             | attention entropy          |

PalmSite exports:

* Per-residue embedding vectors for the predicted catalytic span
* Final attention weights
* Metadata (chunk ID, absolute/relative positions, span method)

---

## **Environment variables**

* `ESM_FORGE_TOKEN` — token for Forge when using `-b 6b`
* `PALMSITE_MODEL_ID` — override default HF repo
* `PALMSITE_MODEL_REV` — optional model revision

---

Version: **0.1.2**

---

## **Citation**
(Coming soon.)
