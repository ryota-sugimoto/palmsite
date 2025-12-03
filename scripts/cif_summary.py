#!/usr/bin/env python3
"""
List entities and chains (struct_asym) with types and lengths from an mmCIF file.

Usage:
    python list_entities_chains.py 1HI0.cif
"""

import argparse
import textwrap
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def as_list(mmcif_dict, key, default=None):
    """Get value for `key` from MMCIF2Dict and normalize to a list."""
    if default is None:
        default = []
    if key not in mmcif_dict:
        return default
    value = mmcif_dict[key]
    # MMCIF2Dict returns a list for looped items, scalar for singletons
    if isinstance(value, list):
        return value
    else:
        return [value]


def clean_seq(seq):
    """Normalize one-letter sequence and return cleaned sequence and its length."""
    if seq is None:
        return None, None
    # Remove whitespace and line breaks
    s = "".join(seq.split())
    if not s or s in (".", "?"):
        return None, None
    return s, len(s)


def build_entity_table(mmcif_dict):
    """
    Build a dict:
      entity_id -> {
         "type": polymer/non-polymer/water,
         "description": str,
         "poly_type": polypeptide(L)/DNA/RNA/... or None,
         "length": int or None
      }
    """
    entity_ids = as_list(mmcif_dict, "_entity.id")
    entity_types = as_list(mmcif_dict, "_entity.type")
    entity_descs = as_list(mmcif_dict, "_entity.pdbx_description")

    # Align lengths
    n_entities = len(entity_ids)
    if len(entity_types) < n_entities:
        entity_types += ["?"] * (n_entities - len(entity_types))
    if len(entity_descs) < n_entities:
        entity_descs += ["?"] * (n_entities - len(entity_descs))

    # Polymer info
    poly_entity_ids = as_list(mmcif_dict, "_entity_poly.entity_id")
    poly_types = as_list(mmcif_dict, "_entity_poly.type")
    poly_seq_can = as_list(mmcif_dict, "_entity_poly.pdbx_seq_one_letter_code_can")
    poly_seq_raw = as_list(mmcif_dict, "_entity_poly.pdbx_seq_one_letter_code")
    poly_num_monomers = as_list(mmcif_dict, "_entity_poly.number_of_monomers")

    entity_poly_info = {}

    for i, eid in enumerate(poly_entity_ids):
        poly_type = poly_types[i] if i < len(poly_types) else "?"
        seq = None
        if i < len(poly_seq_can):
            seq = poly_seq_can[i]
        elif i < len(poly_seq_raw):
            seq = poly_seq_raw[i]

        cleaned_seq, seq_len = clean_seq(seq)

        # If number_of_monomers is provided, use it as length (more robust)
        length = None
        if i < len(poly_num_monomers):
            nm = poly_num_monomers[i]
            if nm not in (".", "?", None):
                try:
                    length = int(nm)
                except ValueError:
                    length = None

        if length is None:
            length = seq_len

        entity_poly_info[eid] = {
            "poly_type": poly_type,
            "length": length,
            "sequence": cleaned_seq,
        }

    # Combine entity-level and polymer info
    entity_info = {}
    for eid, etype, desc in zip(entity_ids, entity_types, entity_descs):
        poly_info = entity_poly_info.get(eid, {})
        entity_info[eid] = {
            "type": etype,
            "description": desc,
            "poly_type": poly_info.get("poly_type"),
            "length": poly_info.get("length"),
            "sequence": poly_info.get("sequence"),
        }

    return entity_info


def build_chain_table(mmcif_dict, entity_info):
    """
    Build a list of chain records based on _struct_asym.
    Each record is a dict:
      {
        "chain_id": str,
        "entity_id": str,
        "entity_type": str,
        "description": str,
        "poly_type": str or None,
        "length": int or None
      }
    """
    asym_ids = as_list(mmcif_dict, "_struct_asym.id")
    asym_entity_ids = as_list(mmcif_dict, "_struct_asym.entity_id")

    chains = []

    for asym_id, eid in zip(asym_ids, asym_entity_ids):
        ent = entity_info.get(eid, {})
        chains.append({
            "chain_id": asym_id,
            "entity_id": eid,
            "entity_type": ent.get("type", "?"),
            "description": ent.get("description", "?"),
            "poly_type": ent.get("poly_type"),
            "length": ent.get("length"),
        })

    return chains


def print_entity_summary(entity_info):
    print("=== Entities ===")
    header = "{:<8} {:<12} {:<20} {:>8}  {}".format(
        "ID", "Type", "PolymerType", "Length", "Description"
    )
    print(header)
    print("-" * len(header))
    for eid in sorted(entity_info.keys(), key=lambda x: int(x) if x.isdigit() else x):
        ent = entity_info[eid]
        etype = ent["type"]
        poly_type = ent["poly_type"] or "-"
        length = ent["length"] if ent["length"] is not None else "-"
        desc = ent["description"]
        # Shorten long descriptions slightly
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print("{:<8} {:<12} {:<20} {:>8}  {}".format(
            eid, etype, poly_type, length, desc
        ))
    print()


def print_chain_summary(chains):
    print("=== Chains (struct_asym) ===")
    header = "{:<8} {:<8} {:<12} {:<20} {:>8}  {}".format(
        "Chain", "Entity", "EntType", "PolymerType", "Length", "Description"
    )
    print(header)
    print("-" * len(header))
    for ch in chains:
        chain_id = ch["chain_id"]
        eid = ch["entity_id"]
        etype = ch["entity_type"]
        poly_type = ch["poly_type"] or "-"
        length = ch["length"] if ch["length"] is not None else "-"
        desc = ch["description"]
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print("{:<8} {:<8} {:<12} {:<20} {:>8}  {}".format(
            chain_id, eid, etype, poly_type, length, desc
        ))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="List entities and chains from an mmCIF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              python list_entities_chains.py 1HI0.cif
            """
        ),
    )
    parser.add_argument("cif", help="Input mmCIF file")
    args = parser.parse_args()

    mmcif_dict = MMCIF2Dict(args.cif)

    entity_info = build_entity_table(mmcif_dict)
    chains = build_chain_table(mmcif_dict, entity_info)

    print_entity_summary(entity_info)
    print_chain_summary(chains)


if __name__ == "__main__":
    main()

