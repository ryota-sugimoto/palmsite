#!/usr/bin/env python3
"""
Convert PalmSite per-residue weights (JSON)
into B-factors in an mmCIF structure.

- Sets ALL atoms to B-factor = 0.0 first.
- For one or more (entity_id, AUTH_chain_id) pairs, residues that can be
  mapped to PalmSite positions get scaled weights as B-factors.
- All other atoms remain at 0.0.

IMPORTANT:
  The chain IDs here are *auth_asym_id* (the ones you see in PyMOL / PDB),
  e.g. for 1HI0 the polymer is entity 2 on chains P, Q, R.

Non-linear scaling:
  We normalize weights to [0,1] and then apply a power gamma:

      w_norm = (w - w_min) / (w_max - w_min)
      w_boost = w_norm ** gamma      # 0 < gamma <= 1 recommended
      B = 100 * w_boost

  gamma < 1 boosts small/mid weights but keeps 0 -> 0 and 1 -> 1.
"""

import argparse
import json
from collections import defaultdict

from Bio.PDB import MMCIFParser, MMCIFIO
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

    if len(w) == 0:
        raise ValueError("Empty weight vector in JSON")

    return w


# ---------------------------------------------------------
# Build mapping: (auth_chain, auth_seq_id) → label_seq_id
# for one or more (entity_id, auth_chain_id) pairs.
# ---------------------------------------------------------
def build_auth_to_seq_map_multi(cif_path, targets):
    """
    Parameters
    ----------
    cif_path : str
        Path to mmCIF file.
    targets : list[tuple[str, str]]
        List of (entity_id, auth_chain_id) pairs, e.g. [("2","P"), ("2","Q")].

    Returns
    -------
    mapping : dict[(str, int), int]
        Dict mapping (auth_chain_id, auth_seq_id) -> label_seq_id (1-based).
    per_target_counts : dict[(str,str), int]
        Number of distinct auth_seq_ids observed for each (entity, auth_chain).
    """
    mmcif = MMCIF2Dict(cif_path)

    label_entity_id = mmcif["_atom_site.label_entity_id"]
    auth_asym_id    = mmcif["_atom_site.auth_asym_id"]
    label_seq_id    = mmcif["_atom_site.label_seq_id"]
    auth_seq_id     = mmcif["_atom_site.auth_seq_id"]

    mapping = {}
    per_target_res = defaultdict(set)

    targets_set = set(targets)

    for e, auth_ch, lseq, aseq in zip(
        label_entity_id,
        auth_asym_id,
        label_seq_id,
        auth_seq_id,
    ):
        if (e, auth_ch) not in targets_set:
            continue
        if lseq in (".", "?", None) or aseq in (".", "?", None):
            continue

        aseq_i = int(aseq)
        lseq_i = int(lseq)

        key = (auth_ch, aseq_i)
        mapping[key] = lseq_i
        per_target_res[(e, auth_ch)].add(aseq_i)

    # Warn / report for each requested target
    per_target_counts = {}
    for t in sorted(targets_set):
        count = len(per_target_res.get(t, set()))
        per_target_counts[t] = count

    if not mapping:
        raise RuntimeError(
            "No residues found for any of the requested (entity, auth_chain) pairs. "
            "Double-check entity IDs and chain IDs (auth_asym_id) in the mmCIF."
        )

    return mapping, per_target_counts


# ---------------------------------------------------------
# Scale weights to 0–100 for B-factor, with non-linear gamma
# ---------------------------------------------------------
def scale_weights(weights, gamma=0.5):
    """
    Scale raw weights to [0, 100] with a power-law non-linearity.

    Parameters
    ----------
    weights : list[float]
    gamma   : float
        1.0  -> linear scaling (original behavior).
        <1.0 -> boosts mid/small values (recommended 0.3–0.7).

    Returns
    -------
    scaled : list[float]
        Same length as weights, in range [0, 100].
    """
    if not weights:
        return []

    w_min = min(weights)
    w_max = max(weights)

    # All weights identical -> just use a constant value
    if w_max == w_min:
        return [50.0] * len(weights)

    # Clamp gamma to a reasonable positive range
    if gamma <= 0:
        gamma = 1.0

    scaled = []
    denom = (w_max - w_min)

    for w in weights:
        # normalize to [0,1]
        w_norm = (w - w_min) / denom
        if w_norm < 0.0:
            w_norm = 0.0
        elif w_norm > 1.0:
            w_norm = 1.0

        # power-law boost
        w_boost = w_norm ** gamma

        scaled.append(100.0 * w_boost)

    return scaled


# ---------------------------------------------------------
# Apply B-factors and write output mmCIF
# ---------------------------------------------------------
def apply_bfactors(cif_path, weights, mapping, target_chains, out_path, gamma):
    """
    Parameters
    ----------
    cif_path : str
    weights : list[float]
        PalmSite weights, one per sequence position (1-based index).
    mapping : dict[(str, int), int]
        (auth_chain_id, auth_seq_id) -> label_seq_id (1-based index into weights).
    target_chains : list[str]
        Auth chain IDs for coloring (e.g. ["P","Q","R"]).
    out_path : str
    gamma : float
        Power for non-linear scaling (see scale_weights).
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)

    # 1) Set ALL atoms' B-factors to 0.0 (all chains, all entities)
    for model in structure:
        for ch in model:
            for residue in ch:
                for atom in residue:
                    atom.set_bfactor(0.0)

    # 2) Apply PalmSite weights to all requested chains in all models
    scaled = scale_weights(weights, gamma=gamma)
    target_chains_set = set(target_chains)

    missing_per_chain = defaultdict(set)
    used_residues_per_chain = defaultdict(set)

    for model in structure:
        for chain in model:
            cid = chain.id
            if cid not in target_chains_set:
                continue

            for residue in chain:
                hetflag, auth_seq, icode = residue.id

                # Skip HETATM / ligands / waters
                if hetflag != " ":
                    continue

                key = (cid, auth_seq)
                if key not in mapping:
                    missing_per_chain[cid].add(auth_seq)
                    continue

                seq_index = mapping[key]  # 1-based
                idx0 = seq_index - 1      # 0-based

                if not (0 <= idx0 < len(scaled)):
                    missing_per_chain[cid].add(auth_seq)
                    continue

                b = scaled[idx0]
                used_residues_per_chain[cid].add(auth_seq)

                for atom in residue:
                    atom.set_bfactor(b)

    # Print a small summary
    print("Coloring summary (per auth chain):")
    for cid in sorted(target_chains_set):
        used = sorted(used_residues_per_chain.get(cid, []))
        missing = sorted(missing_per_chain.get(cid, []))
        print(
            f"  Chain {cid}: "
            f"{len(used)} residues colored, "
            f"{len(missing)} residues left at 0.0"
        )

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(out_path)

    print(f"✔ Wrote colored mmCIF → {out_path}")
    print(f"   Chains colored (auth_asym_id): {', '.join(sorted(target_chains_set))}")
    print("   All other chains/entities: B-factor = 0.0")


# ---------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------
def parse_targets(args):
    """
    Parse target entity/chain pairs.

    Priority:
      1) If --target is given, use all of those (ENTITY:CHAIN, CHAIN is auth_asym_id).
      2) Otherwise, use --entity and --chain as a single pair.
    """
    targets = []

    if args.target:
        for spec in args.target:
            try:
                ent, ch = spec.split(":")
            except ValueError:
                raise ValueError(
                    f"Invalid --target format '{spec}', expected ENTITY:CHAIN (e.g. 2:P)"
                )
            targets.append((ent, ch))
    else:
        targets.append((args.entity, args.chain))

    return targets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Color an mmCIF structure by PalmSite weights (writes B-factors). "
            "CHAIN here is auth_asym_id (what you see in PyMOL / PDB)."
        )
    )

    parser.add_argument(
        "--json",
        required=True,
        help="PalmSite JSON file (contains w[])",
    )

    parser.add_argument(
        "--cif",
        required=True,
        help="Input mmCIF structure",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output mmCIF with modified B-factors",
    )

    # Backward-compatible single entity/chain (auth_asym_id)
    parser.add_argument(
        "--entity",
        default="1",
        help="Entity ID to use (default: 1, ignored if --target is given)",
    )

    parser.add_argument(
        "--chain",
        default="A",
        help="AUTH chain ID to use (default: A, ignored if --target is given)",
    )

    # New: multiple entity/chain pairs
    parser.add_argument(
        "-t",
        "--target",
        action="append",
        help=(
            "Target pair as ENTITY:CHAIN where CHAIN is auth_asym_id "
            "(e.g. 2:P). Can be given multiple times."
        ),
    )

    # New: non-linear scaling power
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help=(
            "Power for non-linear rescaling of weights (0 < gamma <= 1). "
            "gamma=1.0 -> linear (old behavior); "
            "gamma<1 boosts small/mid weights (default: 0.5)."
        ),
    )

    args = parser.parse_args()

    targets = parse_targets(args)
    target_chains = [ch for (_ent, ch) in targets]

    print("Requested targets (entity, auth_chain):")
    for ent, ch in targets:
        print(f"  entity={ent}, chain={ch}")
    print(f"Using gamma={args.gamma} for non-linear scaling")

    weights = load_palmsite_weights(args.json)
    mapping, per_target_counts = build_auth_to_seq_map_multi(args.cif, targets)

    print("Residues found per (entity, auth_chain) in mmCIF:")
    for (ent, ch), n in sorted(per_target_counts.items()):
        print(f"  entity={ent}, chain={ch}: {n} distinct auth_seq_id")

    apply_bfactors(args.cif, weights, mapping, target_chains, args.out, args.gamma)


if __name__ == "__main__":
    main()

