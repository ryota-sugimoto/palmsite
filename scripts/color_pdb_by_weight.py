#!/usr/bin/env python3
"""
Convert PalmSite per-residue weights (JSON)
into B-factors in a PDB file derived from an mmCIF structure.

Usage:
    palmsite_color.py \
        --json 7DTE_weight.json \
        --cif  7DTE.cif \
        --out  7DTE_attn_colored.pdb
"""

import argparse
import json
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# ---------------------------------------------------------
# Load PalmSite weights
# ---------------------------------------------------------
def load_palmsite_weights(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # Take first (and usually only) chunk
    key = next(iter(data.keys()))
    chunk = data[key]

    w = chunk["w"]
    abs_pos = chunk["abs_pos"]

    if len(w) != len(abs_pos):
        raise ValueError("Mismatch: len(w) != len(abs_pos)")

    return w


# ---------------------------------------------------------
# Build mapping: (chain, auth_seq_id) → label_seq_id
# ---------------------------------------------------------
def build_auth_to_seq_map(cif_path, entity_id, chain_id):
    mmcif = MMCIF2Dict(cif_path)

    label_entity_id = mmcif["_atom_site.label_entity_id"]
    label_asym_id   = mmcif["_atom_site.label_asym_id"]
    label_seq_id    = mmcif["_atom_site.label_seq_id"]
    auth_seq_id     = mmcif["_atom_site.auth_seq_id"]

    mapping = {}

    for e, asym, lseq, aseq in zip(label_entity_id,
                                   label_asym_id,
                                   label_seq_id,
                                   auth_seq_id):
        if e != entity_id:
            continue
        if asym != chain_id:
            continue
        if lseq == "." or aseq == ".":
            continue

        key = (chain_id, int(aseq))
        mapping[key] = int(lseq)

    if not mapping:
        raise RuntimeError(
            f"No residues found for entity={entity_id}, chain={chain_id}"
        )

    return mapping


# ---------------------------------------------------------
# Scale weights to 0–100 for B-factor
# ---------------------------------------------------------
def scale_weights(weights):
    w_min = min(weights)
    w_max = max(weights)

    if w_max == w_min:
        return [50.0] * len(weights)

    return [100.0 * (w - w_min) / (w_max - w_min) for w in weights]


# ---------------------------------------------------------
# Apply B-factors and write output PDB
# ---------------------------------------------------------
def apply_bfactors(cif_path, weights, mapping, chain_id, out_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    scaled = scale_weights(weights)
    missing = []

    for residue in chain:
        hetflag, auth_seq, icode = residue.id

        if hetflag != " ":
            continue  # skip HETATM

        key = (chain_id, auth_seq)
        if key not in mapping:
            missing.append(auth_seq)
            continue

        seq_index = mapping[key]  # 1-based
        idx0 = seq_index - 1      # 0-based

        if not (0 <= idx0 < len(scaled)):
            missing.append(auth_seq)
            continue

        b = scaled[idx0]

        for atom in residue:
            atom.set_bfactor(b)

    if missing:
        print("Warning: Missing mapping for auth_seq_ids:", sorted(set(missing)))

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)

    print(f"✔ Wrote colored PDB → {out_path}")


# ---------------------------------------------------------
# Main (argparse)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Color a structure by PalmSite weights (writes B-factors)."
    )

    parser.add_argument("--json", required=True,
                        help="PalmSite JSON file (contains w[])")

    parser.add_argument("--cif", required=True,
                        help="Input mmCIF structure")

    parser.add_argument("--out", required=True,
                        help="Output PDB with B-factors")

    parser.add_argument("--entity", default="1",
                        help="Entity ID to use (default: 1)")

    parser.add_argument("--chain", default="A",
                        help="Chain ID to use (default: A)")

    args = parser.parse_args()

    weights = load_palmsite_weights(args.json)
    mapping = build_auth_to_seq_map(args.cif, args.entity, args.chain)
    apply_bfactors(args.cif, weights, mapping, args.chain, args.out)


if __name__ == "__main__":
    main()

