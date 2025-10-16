# tests/test_cli_smoke.py
from __future__ import annotations

import io
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

# Import the CLI entrypoint
from palmsite.cli import main as palmsite_cli


def _write_dummy_h5(h5_path: Path, base_id: str = "toy") -> None:
    """
    Create the minimal embeddings.h5 structure PalmSite expects:
      /items/<chunk_id>/{emb,mask,seq} + attrs including d_model, orig_aa_start, orig_aa_len
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        g_root = h5.create_group("items")
        # One chunk item that looks like the real embedder output
        chunk_id = f"{base_id}|chunk_0001_of_0001|aa_000000_000012"
        g = g_root.create_group(chunk_id)
        L, D = 12, 8
        emb = np.zeros((L, D), dtype=np.float32)  # small token-wise embedding
        mask = np.ones((L,), dtype=bool)
        g.create_dataset("emb", data=emb)
        g.create_dataset("mask", data=mask)
        dt = h5py.special_dtype(vlen=str)
        g.create_dataset("seq", data="M" * L, dtype=dt)
        # attrs used by the predictor
        g.attrs["d_model"] = D
        g.attrs["orig_aa_start"] = 0
        g.attrs["orig_aa_len"] = L


@pytest.fixture
def dummy_embed(monkeypatch, tmp_path):
    """
    Replace the heavy embedder with a no-op that writes a tiny, valid HDF5 to the requested path.
    """

    # We need to monkeypatch the function the CLI calls:
    # palmsite.embed_shim.embed_fastas_to_h5(fasta_path=..., h5_path=..., backbone=..., ...)
    def _fake_embed(fasta_path: str, h5_path: str, **kwargs):
        _write_dummy_h5(Path(h5_path), base_id="toy")

    monkeypatch.setattr("palmsite.cli.embed_fastas_to_h5", _fake_embed)
    return _fake_embed


@pytest.fixture
def dummy_predict(monkeypatch):
    """
    Replace the heavy predictor with a tiny writer that emits one deterministic GFF row to the stream.
    """

    # CLI calls: palmsite.infer_simple.predict_to_gff(embeddings_h5, backbone, model_id, revision, min_p, out_stream)
    def _fake_predict(embeddings_h5: str, backbone: str, model_id, revision, min_p: float, out_stream):
        # Confirm the H5 exists and has the expected minimal structure
        assert Path(embeddings_h5).exists(), "embeddings.h5 missing"
        with h5py.File(embeddings_h5, "r") as h5:
            assert "items" in h5
        # Emit a simple, valid GFF3 header + one feature above min_p
        out_stream.write("##gff-version 3\n")
        out_stream.write("toy\tPalmSite\tRdRP_domain\t3\t9\t0.900000\t.\t.\tID=toy;P=0.900000\n")

    monkeypatch.setattr("palmsite.cli.predict_to_gff", _fake_predict)
    return _fake_predict


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sid, seq in records:
            f.write(f">{sid}\n")
            f.write(seq + "\n")


def test_stdout_when_no_output_specified(tmp_path, dummy_embed, dummy_predict):
    """
    When -o/--gff-out is not provided, GFF should go to stdout.
    Also verifies short option -p works.
    """
    fa = tmp_path / "a.faa"
    _write_fasta(fa, [("seq1", "M" * 20)])

    runner = CliRunner()
    # Use backbone 300m to exercise option parsing (the fake embed ignores it)
    result = runner.invoke(palmsite_cli, ["-p", "0.5", "-b", "300m", str(fa)])

    assert result.exit_code == 0, result.output
    assert "##gff-version 3" in result.output
    assert "\tPalmSite\tRdRP_domain\t" in result.output


def test_file_output_and_multi_fasta(tmp_path, dummy_embed, dummy_predict):
    """
    When -o/--gff-out is provided, write there.
    Also verifies multiple FASTA inputs and long options.
    """
    fa1 = tmp_path / "a.faa"
    fa2 = tmp_path / "b.faa"
    _write_fasta(fa1, [("A", "M" * 10)])
    _write_fasta(fa2, [("B", "M" * 15)])

    out_gff = tmp_path / "result.gff"

    runner = CliRunner()
    result = runner.invoke(
        palmsite_cli,
        [
            "--min-p", "0.50",
            "--gff-out", str(out_gff),
            "--backbone", "600m",
            str(fa1),
            str(fa2),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_gff.exists(), "GFF output file not created"
    txt = out_gff.read_text(encoding="utf-8")
    assert txt.startswith("##gff-version 3")
    assert "\tPalmSite\tRdRP_domain\t" in txt


def test_short_and_long_flags_equivalence(tmp_path, dummy_embed, dummy_predict):
    """
    Ensure short (-o/-p/-b) and long (--gff-out/--min-p/--backbone) flags both work.
    """
    fa = tmp_path / "c.faa"
    _write_fasta(fa, [("C", "M" * 12)])
    out1 = tmp_path / "x.gff"
    out2 = tmp_path / "y.gff"

    runner = CliRunner()
    r1 = runner.invoke(palmsite_cli, ["-p", "0.6", "-o", str(out1), "-b", "6b", str(fa)])
    r2 = runner.invoke(palmsite_cli, ["--min-p", "0.6", "--gff-out", str(out2), "--backbone", "6b", str(fa)])

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert out1.exists() and out2.exists()
    assert out1.read_text() == out2.read_text()
