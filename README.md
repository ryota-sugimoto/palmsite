# PalmSite — RdRP catalytic center predictor

PalmSite is a simple, fast command-line tool that predicts the **RNA-dependent RNA polymerase (RdRP) catalytic center** from protein FASTA and outputs **GFF3**.

## Highlights

- **One command** from FASTA → GFF3: `palmsite <fasta ...>`
- **High precision and recall** (internal benchmarks):

| Backbone (ESM-C) | Positives vs. Negatives | Positives vs. Rest |
|---|---:|---:|
| 6b   | 0.9998 | 0.9848 |
| 600m | 0.9992 | 0.9687 |
| 300m | 0.9991 | 0.9755 |

- Detects **distant homologs** (e.g., HSRV RdRP).
- Clear **progress logging** and fast **batched embedding** (local 300m/600m or Forge 6B).

## Installation

```bash
pip install palmsite
```

## Quickstart

```bash
# Basic (local 600m is default backbone)
palmsite examples/zikavirus_proteins.fasta

# Quiet mode (only errors)
palmsite -q examples/zikavirus_proteins.fasta

# Raise the reporting threshold
palmsite -p 0.9 examples/zikavirus_proteins.fasta

# Use 6B (Forge); requires a token
palmsite -b 6b -k <FORGE_TOKEN> examples/zikavirus_proteins.fasta
# or export ESM_FORGE_TOKEN and omit -k
```

Notes:
- `-b/--backbone` chooses the embedding model: **300m**, **600m** (local), or **6b** (Forge/cloud).

## Command-line usage

```
Usage: palmsite [OPTIONS] [FASTAS]...

  PalmSite — RdRP catalytic center predictor. Usage: palmsite -p 0.5 [-o
  result.gff] [options] <fasta ...>

Options:
  --version                      Show the version and exit.
  -o, --gff-out TEXT             Write GFF3 to this path; default: stdout if
                                 omitted
  -p, --min-p FLOAT              Minimum probability to include a feature in
                                 GFF  [default: 0.5]
  -b, --backbone [300m|600m|6b]  Embedding backbone & size: '300m' (fast,
                                 local), '600m' (balanced, local), '6b'
                                 (highest quality via ESM Forge; requires
                                 --token or ESM_FORGE_TOKEN).  [default: 600m]
  -m, --model-id TEXT            Hugging Face model repo (default via
                                 PALMSITE_MODEL_ID env or palmsite/<backbone>)
  -d, --device [auto|cpu|cuda]   Device for local ESM-C (ignored for 6B Forge)
                                 [default: auto]
  -k, --token TEXT               Forge token (required for 6B if not set in
                                 ESM_FORGE_TOKEN)
  -t, --tmp-dir TEXT             Optional working directory for temp files
  -q, --quiet                    Reduce non-error logs
  -v, --verbose                  Verbose logs (DEBUG level; overrides -q)
  --keep-tmp                     Keep temporary files (sanitized FASTA &
                                 embeddings.h5) for debugging
```

### About `-b/--backbone`

- **300m** – fast local ESM-C. Good for CPU/GPU prototyping.
- **600m** – balanced local ESM-C. Better quality; still lightweight.
- **6b** – highest quality via **ESM Forge** (cloud); requires `-k <token>` or `ESM_FORGE_TOKEN`.

## What PalmSite does

1. **Sanitize & merge FASTA**  
   Replaces unusual residues with `X`, drops sequences if too many fixes were needed, and writes one merged FASTA.
2. **Embed sequences**  
   - Launches the embedding engine (batched, token-aware micro-batching; visible progress/ETA).
   - Backends:
     - Local ESM-C (**300m/600m**) via Hugging Face.
     - Forge (**6B**) via the ESM SDK (`ESM3ForgeInferenceClient`).
3. **Predict → GFF3**  
   Loads the checkpoint from Hugging Face, computes RdRP probabilities and spans, aggregates per protein, and writes **GFF3**.

## Output

- **GFF3** (stdout or `-o`): one feature per protein (catalytic center span). Attributes include `P`, `sigma`, original length, and the chunk used.

## Environment variables

- `ESM_FORGE_TOKEN` — Forge API token for `-b 6b` (alternative to `-k`).

## Project structure (user-side)

- `cli.py` — top-level command: sanitize → embed → predict.
- `embed_shim.py` — launches the embedding engine in a subprocess.
- `_embed_impl.py` — embedding engine (batching, progress, HDF5 writer, Forge/local backends).
- `infer_simple.py` — simple driver to produce GFF from embeddings.
- `_predict_impl.py` — full predictor (model, dataset, collate).
- `hf.py` — Hugging Face weight resolution.
- `sanitize.py` — FASTA cleaner/merger.
- `__init__.py` — version. Current: **0.1.0**.

## Version

**0.1.0**
