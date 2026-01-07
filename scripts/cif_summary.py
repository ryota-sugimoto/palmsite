#!/usr/bin/env python3
"""
List entities and chains (struct_asym) with types and lengths from an mmCIF file.

Reports:
  - Entity summary (ID, type, polymer type, length, description)
  - Chain summary with BOTH:
      * label_asym_id (struct_asym.id)
      * auth_asym_id(s) (from _atom_site.auth_asym_id)

Usage:
    python cif_summary.py 1HI0.cif
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
    Build a list of chain records based on _struct_asym and _atom_site.

    Each record is a dict:
      {
        "label_chain_id": str,        # struct_asym.id == label_asym_id
        "auth_chain_ids": str,        # comma-joined unique auth_asym_id(s)
        "entity_id": str,
        "entity_type": str,
        "description": str,
        "poly_type": str or None,
        "length": int or None
      }
    """
    # Chains from _struct_asym (label-based view)
    asym_ids = as_list(mmcif_dict, "_struct_asym.id")         # label_asym_id
    asym_entity_ids = as_list(mmcif_dict, "_struct_asym.entity_id")

    # Build mapping: label_asym_id -> set(auth_asym_id) from _atom_site
    label_asym_atom = as_list(mmcif_dict, "_atom_site.label_asym_id")
    auth_asym_atom = as_list(mmcif_dict, "_atom_site.auth_asym_id")

    label_to_auth = {}
    if label_asym_atom and auth_asym_atom:
        for lab, auth in zip(label_asym_atom, auth_asym_atom):
            if lab not in label_to_auth:
                label_to_auth[lab] = set()
            label_to_auth[lab].add(auth)

    chains = []

    for asym_id, eid in zip(asym_ids, asym_entity_ids):
        ent = entity_info.get(eid, {})
        auth_set = label_to_auth.get(asym_id, set())
        if not auth_set:
            auth_str = "?"
        else:
            # Sort for deterministic output
            auth_str = ",".join(sorted(auth_set))

        chains.append({
            "label_chain_id": asym_id,
            "auth_chain_ids": auth_str,
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
    header = "{:<8} {:<10} {:<8} {:<12} {:<20} {:>8}  {}".format(
        "Label", "Auth", "Entity", "EntType", "PolymerType", "Length", "Description"
    )
    print(header)
    print("-" * len(header))
    for ch in chains:
        label_id = ch["label_chain_id"]
        auth_ids = ch["auth_chain_ids"]
        eid = ch["entity_id"]
        etype = ch["entity_type"]
        poly_type = ch["poly_type"] or "-"
        length = ch["length"] if ch["length"] is not None else "-"
        desc = ch["description"]
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print("{:<8} {:<10} {:<8} {:<12} {:<20} {:>8}  {}".format(
            label_id, auth_ids, eid, etype, poly_type, length, desc
        ))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="List entities and chains from an mmCIF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              python cif_summary.py 1HI0.cif

            Notes:
              - 'Label' = _struct_asym.id (label_asym_id)
              - 'Auth'  = _atom_site.auth_asym_id (author chain ID, as seen in PyMOL)
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

