# **PalmSite — RdRP catalytic center predictor**

PalmSite is a fast command-line tool that predicts the **RNA-dependent RNA polymerase (RdRP) catalytic center** from protein FASTA and outputs **GFF3**.
As of **v0.2.0**, PalmSite can also optionally output **per-residue attention weights** and **span parameters** in **JSON**.

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

* **New:** compact pooled internal-backbone vectors for zero-shot analysis:

  ```bash
  palmsite --pooled-json pooled_panels.json <fasta>
  ```

* **New:** per-residue final PalmSite backbone vectors for embedding-based residue alignment:

  ```bash
  palmsite --backbone-json backbone_vectors.json <fasta>
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

# Use a local PalmSite checkpoint instead of Hugging Face weights
palmsite --model-pt runs/debug/model_best.pt examples/hsrv_proteins.fasta
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

## **NEW: Pooled backbone panels for zero-shot test**

PalmSite can now export compact per-chunk pooled vectors from the **final PalmSite backbone layer** `H` rather than from the raw input embedding. The main recommended representation is:

```text
pools.backbone.span_attn_norm
```

This vector is the final backbone embedding averaged over the predicted catalytic span `S_idx:E_idx`, using final attention weights renormalized inside that span, then L2-normalized by default.

```bash
palmsite \
  -o result.gff \
  --attn-json attention_details.json \
  --pooled-json pooled_panels.json \
  examples/myproteins.fasta
```

For each chunk, `pooled_panels.json` contains:

```json
{
  "chunk_id": {
    "base_id": "original_sequence_id",
    "is_best_base_chunk": true,
    "P": 0.99,
    "S_idx": 100,
    "E_idx": 180,
    "pools": {
      "backbone": {
        "full_mean": [...],
        "full_attn_norm": [...],
        "span_mean": [...],
        "span_attn_norm": [...],
        "topk_attn_norm": [...],
        "nonspan_mean": [...]
      }
    },
    "pool_meta": {
      "schema": "palmsite_pooled_panels.v1",
      "backbone_dim": 256,
      "l2_normalized": true
    }
  }
}
```

Useful options:

```bash
# Also embed the same vectors inside each attention JSON entry
palmsite --attn-json attention.json --include-pools-in-attn-json <fasta>

# Add raw ESM-C input-embedding control panels; this makes JSON much larger
palmsite --pooled-json pooled.json --pool-include-input <fasta>

# Change the top-k panel size or disable L2 normalization
palmsite --pooled-json pooled.json --pool-top-k 64 --pool-no-l2 <fasta>
```

For one vector per original protein in downstream clustering, keep only records with:

```text
is_best_base_chunk == true
```

---

## **NEW: Per-residue PalmSite backbone vectors for embedding alignment**

For residue-residue embedding alignment, PalmSite can export the full token-level internal representation:

```bash
palmsite \
  -o result.gff \
  --backbone-json backbone_vectors.json \
  examples/myproteins.fasta
```

By default, this writes vectors only for the predicted catalytic span to keep the JSON smaller. Use `--backbone-json-scope full` to export all valid residues in each embedded chunk.

Each entry in `backbone_vectors.json` contains:

```json
{
  "chunk_id": {
    "base_id": "original_sequence_id",
    "is_best_base_chunk": true,
    "P": 0.99,
    "S_idx": 100,
    "E_idx": 180,
    "scope": "span",
    "local_pos": [100, 101, 102],
    "abs_pos": [500, 501, 502],
    "w": [... final PalmSite attention weights for exported residues ...],
    "vectors": [[... 256-dim PalmSite backbone H vector ...]]
  }
}
```

Useful options:

```bash
# Export all valid residues instead of only the predicted catalytic span
palmsite --backbone-json backbone_vectors.json --backbone-json-scope full <fasta>

# Export only chunks above a PalmSite probability threshold
palmsite --backbone-json backbone_vectors.json --backbone-json-min-p 0.5 <fasta>

# Also include raw ESM-C token vectors as a control representation
palmsite --backbone-json backbone_vectors.json --backbone-json-include-input <fasta>
```

This JSON can become very large because each exported residue stores a 256-dimensional vector. For embedding alignment focused on the catalytic core, the default `span` scope is usually the practical choice.

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
  -m, --model-id TEXT             HF model repo for PalmSite weights (default: ryota-sugimoto/palmsite)
  --model-pt, --checkpoint PATH    Local PalmSite checkpoint (.pt); overrides HF download
  -d, --device [auto|cpu|cuda]    Device for local models (ignored for 6b)
  -k, --token TEXT                Forge token for 6B (or set ESM_FORGE_TOKEN)
  -t, --tmp-dir PATH              Temp directory (default: auto-created)
  -q, --quiet                     Suppress logs
  -v, --verbose                   Debug logs (overrides quiet)
  --keep-tmp                      Keep temp files (sanitized FASTA + per-batch embeddings)
  --attn-json PATH                Write per-residue attention JSON (can be large)
  --pooled-json PATH              Write compact pooled backbone vector panels
  --backbone-json PATH            Write per-residue PalmSite backbone H vectors
  --backbone-json-scope [span|full]
                                  Export predicted span only or full valid chunk [default: span]
  --backbone-json-min-p FLOAT     Minimum P for backbone-vector JSON entries [default: 0.0]
  --backbone-json-include-input   Also include raw ESM-C token vectors as controls
  --include-pools-in-attn-json    Embed pooled panels inside each attention JSON entry
  --pool-include-input            Also include raw ESM-C input-embedding control panels
  --pool-top-k INTEGER            Number of residues for top-k attention panel [default: 32]
  --pool-no-l2                    Disable L2 normalization of pooled vectors
  --micro-batch-seqs INTEGER      Micro-batch size in number of sequences
  --micro-batch-tokens INTEGER    Micro-batch size cap in ~tokens (sum(len(seq)+2))
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

**Streaming micro-batches (v0.2.0+)**: the CLI runs embedding and prediction in small micro-batches, emitting GFF3 rows incrementally and deleting each temporary embedding HDF5 right after it is consumed (unless you pass `--keep-tmp`). This avoids large peak disk usage for big FASTA inputs.

Tune with:
* `--micro-batch-tokens` (default: ~80k for local backbones, ~120k for 6b)
* `--micro-batch-seqs` (optional hard cap on number of sequences per batch)

### 3. **Predict RdRP domains**

Prediction code lives in:

* `_predict_impl.py` (full engine with CSV, GFF3, HDF5 export, and JSON export) 
* `infer_simple.py` (minimal GFF3 generator, now with JSON support) 

Outputs include:

* **GFF3** spans
* **JSON** with attention maps
* **Pooled final-backbone vector panels** for clustering/taxonomy comparisons

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

---

## **Environment variables**

* `ESM_FORGE_TOKEN` — token for Forge when using `-b 6b`
* `PALMSITE_MODEL_ID` — override default HF repo
* `PALMSITE_MODEL_REV` — optional model revision

When `--model-pt` is provided, PalmSite loads that local `.pt` checkpoint directly and does not download PalmSite weights from Hugging Face. The selected `--backbone` should still match the checkpoint you trained.

---

Version: **0.2.0**

---

## **Citation**
(Coming soon.)
