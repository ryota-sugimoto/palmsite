from __future__ import annotations
import os, sys, tempfile, shutil, click
from pathlib import Path

from .sanitize import sanitize_and_merge_fastas
from .embed_shim import embed_fastas_to_h5
from .infer_simple import predict_to_gff
from . import __version__

@click.command(context_settings=dict(help_option_names=["-h","--help"]))
@click.version_option(__version__, prog_name="palmsite")
@click.option("-o","--gff-out", default=None,
              help="Write GFF3 to this path; default: stdout if omitted")
@click.option("-p","--min-p", type=float, default=0.50, show_default=True,
              help="Minimum probability to include a feature in GFF")
@click.option("-b","--backbone", type=click.Choice(["300m","600m","6b"]), default="600m", show_default=True,
              help=("Embedding backbone & size: "
                    "'300m' (fast, local), '600m' (balanced, local), '6b' (highest quality via ESM Forge; "
                    "requires --token or ESM_FORGE_TOKEN)."))
@click.option("-m","--model-id", default=None,
              help="Hugging Face model repo (default via PALMSITE_MODEL_ID env or palmsite/<backbone>)")
@click.option("-r","--revision", default=None, hidden=True,
              help="(Deprecated) HF tag or commit (e.g., v1). Prefer defaults or PALMSITE_MODEL_REV.")
@click.option("-d","--device", type=click.Choice(["auto","cpu","cuda"]), default="auto", show_default=True,
              help="Device for local ESM-C (ignored for 6B Forge)")
@click.option("-k","--token", default=None,
              help="Forge token (required for 6B if not set in ESM_FORGE_TOKEN)")
@click.option("-t","--tmp-dir", default=None, help="Optional working directory for temp files")
@click.option("-q","--quiet", is_flag=True, help="Reduce non-error logs")
@click.option("-v","--verbose", is_flag=True, help="Verbose logs (DEBUG level; overrides -q)")
@click.option("--keep-tmp", is_flag=True, help="Keep temporary files (sanitized FASTA & embeddings.h5) for debugging")
@click.argument("fastas", nargs=-1, type=click.Path(exists=True))
def main(gff_out, min_p, backbone, model_id, revision, device, token, tmp_dir, quiet, verbose, keep_tmp, fastas):
    """
    PalmSite — RdRP catalytic center predictor.
    Usage: palmsite -p 0.5 [-o result.gff] [options] <fasta ...>
    """
    if not fastas:
        click.echo("No FASTA provided. See `palmsite --help`.", err=True)
        sys.exit(2)

    # Pick log level for embedding engine
    if verbose:
        log_level = "DEBUG"
        quiet = False  # verbose wins
    elif quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="palmsite_"))
    tmp_base.mkdir(parents=True, exist_ok=True)
    merged_fa = tmp_base / "sanitized_merged.fasta"
    emb_h5 = tmp_base / "embeddings.h5"

    # 1) Sanitize & merge
    try:
        if not quiet:
            click.echo("[1/3] Sanitizing & merging FASTA…", err=True)
        n_ok, n_fixed, n_dropped = sanitize_and_merge_fastas(
            [Path(p) for p in fastas], merged_fa, strict_fraction=0.20, quiet=quiet
        )
        if n_ok == 0:
            click.echo("All sequences were invalid after sanitization; aborting.", err=True)
            sys.exit(2)
        if not quiet:
            click.echo(f"Sanitized sequences: kept={n_ok}, fixed={n_fixed}, dropped={n_dropped}", err=True)
    except Exception as e:
        click.echo(f"Sanitization failed: {e}", err=True)
        sys.exit(2)

    # 2) Embed (ESM‑C 300m/600m locally or 6B via Forge)
    try:
        if not quiet:
            click.echo(f"[2/3] Embedding with backbone={backbone}…", err=True)
        embed_fastas_to_h5(
            fasta_path=str(merged_fa),
            h5_path=str(emb_h5),
            backbone=backbone,
            device=device,
            token=token,
            quiet=quiet,
            log_level=log_level,
            progress_every=25,
        )
    except Exception as e:
        click.echo(f"Embedding failed: {e}", err=True)
        sys.exit(2)

    # 3) Infer → 4) GFF
    out_fp = None
    try:
        if not quiet:
            click.echo("[3/3] Predicting and writing GFF…", err=True)
        if gff_out:
            out_fp = open(gff_out, "w", encoding="utf-8")
            stream = out_fp
        else:
            stream = sys.stdout

        predict_to_gff(
            embeddings_h5=str(emb_h5),
            backbone=backbone,
            model_id=model_id,
            revision=revision,
            min_p=float(min_p),
            out_stream=stream,
        )
    except Exception as e:
        click.echo(f"Inference failed: {e}", err=True)
        sys.exit(2)
    finally:
        if out_fp:
            out_fp.close()
        if not tmp_dir and not keep_tmp:
            # cleanup our own temp dir unless user asked to keep it
            shutil.rmtree(tmp_base, ignore_errors=True)
