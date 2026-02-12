#!/usr/bin/env python
"""Inspect CAVEAT pipeline results by writing SDF files.

Usage:
    # From SMILES:
    python scripts/inspect_results.py \
        --db test.db \
        --mol "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5" \
        --replace "CN1CCNCC1" \
        --outdir results/ --top 5

    # From SDF (preserves docked/experimental 3D coordinates):
    python scripts/inspect_results.py \
        --db chembl_5k.db \
        --mol-sdf docked_ligand.sdf \
        --replace "COc1cnnc1" \
        --outdir results/ --top 100

Produces:
    results/
        parent.sdf            — query molecule in 3D
        parent_highlight.sdf  — query molecule with matched atoms as a property
        fragments/
            frag_001.sdf      — each replacement fragment aligned to parent frame
        assembled/
            assembled_001.sdf — each assembled product aligned to parent frame
        all_fragments.sdf     — combined fragment file
        all_assembled.sdf     — combined assembled file
        properties.csv        — property table for all assembled products

All structures are in the same coordinate frame as the parent molecule.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdDistGeom, rdForceFieldHelpers, rdMolDescriptors
from rdkit.Geometry import Point3D

from caveat.database import FragmentDatabase, _find_attachment_points_in_mol
from caveat.fragment import AttachmentPoint
from caveat.query import find_replacements, _find_cut_bonds, ReplacementResult, _strip_brics_labels, _deduplicate_by_core
from caveat.assemble import assemble
from caveat.geometry import (
    align_single_vector, align_two_vectors, apply_transform,
    compute_vector_pair_descriptor,
    refine_rotation_around_axis,
)


# --- Property computation ---

def compute_properties(mol: Chem.Mol) -> dict:
    """Compute standard drug-like molecular properties."""
    return {
        "MW": round(Descriptors.ExactMolWt(mol), 1),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "ArRings": Descriptors.NumAromaticRings(mol),
        "SatRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
    }


def compute_hcapped_properties(smiles: str) -> dict | None:
    """Compute properties on a fragment with dummy atoms replaced by H.

    Used for fast delta estimation before full assembly.
    """
    core = re.sub(r'\[\d+\*\]', '[H]', smiles)
    mol = Chem.MolFromSmiles(core)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return compute_properties(mol)


def build_hcapped_substructure(parent_mol: Chem.Mol, match_atoms: tuple) -> Chem.Mol | None:
    """Extract the matched substructure from parent and cap cut bonds with H.

    Returns a sanitized molecule suitable for property computation.
    """
    match_set = set(match_atoms)
    emol = Chem.RWMol(Chem.Mol())
    idx_map = {}
    for old_idx in sorted(match_set):
        atom = parent_mol.GetAtomWithIdx(old_idx)
        new_idx = emol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        idx_map[old_idx] = new_idx

    for bond in parent_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in match_set and a2 in match_set:
            emol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())

    # Cap cut bonds with H
    for bond in parent_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            matched = a1 if a1 in match_set else a2
            ext = a2 if a1 in match_set else a1
            if parent_mol.GetAtomWithIdx(ext).GetAtomicNum() == 1:
                continue
            h_idx = emol.AddAtom(Chem.Atom(1))
            emol.AddBond(idx_map[matched], h_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(emol)
        return Chem.RemoveHs(emol.GetMol())
    except Exception:
        return None


def passes_filters(props: dict, filters: dict) -> bool:
    """Check if properties pass all absolute filter ranges."""
    for key, (lo, hi) in filters.items():
        if key not in props:
            continue
        val = props[key]
        if val < lo or val > hi:
            return False
    return True


def passes_delta_filters(props: dict, parent_props: dict, delta_filters: dict) -> bool:
    """Check if property deltas (product - parent) pass all delta filter ranges.

    Each entry in delta_filters is key -> (lo, hi) applied to (props[key] - parent_props[key]).
    """
    for key, (lo, hi) in delta_filters.items():
        if key not in props or key not in parent_props:
            continue
        delta = props[key] - parent_props[key]
        # Use small epsilon for float comparisons
        if delta < lo - 1e-9 or delta > hi + 1e-9:
            return False
    return True


DEFAULT_FILTERS = {
    "MW": (100, 750),
    "cLogP": (-3, 7),
    "HBA": (0, 12),
    "HBD": (0, 6),
    "RotBonds": (0, 15),
    "TPSA": (0, 200),
}


def parse_filter_arg(filter_str: str) -> dict:
    """Parse filter string like 'MW:100-700,cLogP:-2-6,HBD:0-5'."""
    filters = {}
    for part in filter_str.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, range_str = part.partition(":")
        key = key.strip()
        range_str = range_str.strip()
        # Handle negative numbers: split on '-' but be careful with negatives
        parts = range_str.split("-")
        if len(parts) == 2:
            lo, hi = float(parts[0]), float(parts[1])
        elif len(parts) == 3 and parts[0] == "":
            lo, hi = -float(parts[1]), float(parts[2])
        else:
            print(f"WARNING: Could not parse filter '{part}', skipping")
            continue
        filters[key] = (lo, hi)
    return filters


def parse_delta_filter_arg(filter_str: str) -> dict:
    """Parse delta filter string.

    Syntax:
        MW<15       — |delta| < 15, i.e. delta in (-15, +15)
        cLogP<0.5   — |delta| < 0.5
        HBA=0       — delta == 0
        ArRings=-1  — delta == -1
        TPSA=any    — no constraint (skip)

    Multiple filters separated by commas: "MW<15,cLogP<0.5,HBA=0,ArRings=-1"
    """
    filters = {}
    for part in filter_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "<" in part:
            key, _, val_str = part.partition("<")
            val = float(val_str.strip())
            filters[key.strip()] = (-val, val)
        elif "=" in part:
            key, _, val_str = part.partition("=")
            val_str = val_str.strip()
            if val_str.lower() == "any":
                continue  # skip — no constraint
            val = float(val_str)
            filters[key.strip()] = (val, val)
        else:
            print(f"WARNING: Could not parse delta filter '{part}', use KEY<VAL or KEY=VAL")
    return filters


# --- Property-first search ---

def find_replacements_property_first(
    db: FragmentDatabase,
    mol_3d: Chem.Mol,
    match_atoms: set[int],
    cut_bonds,
    num_aps: int,
    sub_props: dict,
    delta_filters: dict,
    top_k: int = 200,
) -> list[ReplacementResult]:
    """Find replacements by filtering on property deltas first, then ranking by geometry.

    This bypasses the KDTree geometric search to find fragments that match
    property criteria regardless of geometric distance.
    """
    # 1. Get ALL fragments with the right number of APs
    rows = db.conn.execute(
        """SELECT id, canonical_smiles, num_heavy_atoms, source_count
           FROM fragments WHERE num_attachment_points = ?""",
        (num_aps,),
    ).fetchall()
    print(f"  Total {num_aps}-AP fragments in DB: {len(rows)}")

    # 2. Filter by H-capped property deltas
    property_matches = []
    n_no_props = 0
    for fid, smi, nha, sc in rows:
        frag_hcapped = compute_hcapped_properties(smi)
        if frag_hcapped is None:
            n_no_props += 1
            continue
        if passes_delta_filters(frag_hcapped, sub_props, delta_filters):
            property_matches.append((fid, smi, nha, sc))

    print(f"  Property-matching fragments: {len(property_matches)} "
          f"(skipped {n_no_props} unparseable)")

    if not property_matches:
        return []

    # 3. Compute query's geometric descriptor(s) for distance ranking
    query_descriptors = []
    for conf_idx in range(mol_3d.GetNumConformers()):
        conf = mol_3d.GetConformer(conf_idx)
        if num_aps >= 2:
            for i in range(len(cut_bonds)):
                for j in range(i + 1, len(cut_bonds)):
                    cb1, cb2 = cut_bonds[i], cut_bonds[j]
                    desc = compute_vector_pair_descriptor(
                        conf,
                        dummy1_idx=cb1.external_atom_idx,
                        neighbor1_idx=cb1.matched_atom_idx,
                        dummy2_idx=cb2.external_atom_idx,
                        neighbor2_idx=cb2.matched_atom_idx,
                    )
                    if desc is not None:
                        query_descriptors.append(desc)

    # 4. For each property-matching fragment, compute geometric distance
    frag_ids = [m[0] for m in property_matches]
    # Get vector_pairs for these fragments in bulk
    if frag_ids and query_descriptors:
        placeholders = ",".join("?" * len(frag_ids))
        vp_rows = db.conn.execute(
            f"""SELECT fragment_id, distance, angle1, angle2, dihedral
                FROM vector_pairs
                WHERE fragment_id IN ({placeholders})""",
            frag_ids,
        ).fetchall()

        # Group by fragment_id, keep best distance per fragment
        frag_best_dist = {}
        for frag_id, d, a1, a2, dih in vp_rows:
            for qdesc in query_descriptors:
                # Compute normalized distance (same as KDTree)
                geo_dist = np.sqrt(
                    (d - qdesc.distance) ** 2
                    + ((a1 - qdesc.angle1) / 15.0) ** 2
                    + ((a2 - qdesc.angle2) / 15.0) ** 2
                    + ((dih - qdesc.dihedral) / 30.0) ** 2
                )
                if frag_id not in frag_best_dist or geo_dist < frag_best_dist[frag_id]:
                    frag_best_dist[frag_id] = geo_dist
    else:
        frag_best_dist = {}

    # 5. Build results, using geo_dist if available, else large value
    results = []
    n_no_conformer = 0
    for fid, smi, nha, sc in property_matches:
        if fid in frag_best_dist:
            geo_dist = frag_best_dist[fid]
        else:
            n_no_conformer += 1
            geo_dist = 999.0  # no conformer — rank last but still include

        results.append(ReplacementResult(
            fragment_id=fid,
            smiles=smi,
            geometric_distance=geo_dist,
            num_attachment_points=num_aps,
            num_heavy_atoms=nha,
            source_count=sc,
        ))

    if n_no_conformer:
        print(f"  WARNING: {n_no_conformer} property-matching fragments have no conformers "
              f"(will rank last)")

    # 6. Sort by geometric distance, deduplicate, take top_k
    results.sort(key=lambda r: r.geometric_distance)
    results = _deduplicate_by_core(results)
    print(f"  After dedup: {len(results)} unique core scaffolds")
    return results[:top_k]


# --- 3D helpers ---

def make_3d(mol, n_confs=1, seed=42):
    """Embed a molecule in 3D with hydrogens."""
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    if mol.GetNumConformers() > 0:
        try:
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol


def write_sdf(mol, path, props=None):
    """Write a single molecule to an SDF file."""
    writer = Chem.SDWriter(path)
    if props:
        for k, v in props.items():
            mol.SetProp(k, str(v))
    writer.write(mol)
    writer.close()
    print(f"  Wrote: {path}")


def align_fragment_to_parent(frag_mol, frag_aps, parent_3d, cut_info):
    """Align a fragment's stored 3D conformer to the parent's cut-point vectors.

    Returns a copy of frag_mol with coordinates transformed to the parent frame.
    """
    if frag_mol.GetNumConformers() == 0:
        return frag_mol

    conf = frag_mol.GetConformer(0)
    parent_conf = parent_3d.GetConformer(0)

    # Gather all fragment atom positions
    n_atoms = frag_mol.GetNumAtoms()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])

    n_aps = min(len(frag_aps), len(cut_info))

    if n_aps == 1:
        ap = frag_aps[0]
        ci = cut_info[0]

        frag_neighbor_pos = np.array(list(conf.GetAtomPosition(ap.neighbor_atom_idx)))
        frag_dummy_pos = np.array(list(conf.GetAtomPosition(ap.dummy_atom_idx)))
        # neighbor → internal position, exit vector points toward external
        target_pos = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))
        target_dir_pos = np.array(list(parent_conf.GetAtomPosition(ci["external_idx"])))

        transform = align_single_vector(
            positions, frag_neighbor_pos, frag_dummy_pos,
            target_pos, target_dir_pos,
        )

        # Refine rotation around the bond axis using subsidiary bonds
        aligned_tmp = apply_transform(positions, transform)
        axis = target_dir_pos - target_pos
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            neighbor_atom = frag_mol.GetAtomWithIdx(ap.neighbor_atom_idx)
            frag_other = [n.GetIdx() for n in neighbor_atom.GetNeighbors()
                          if n.GetIdx() != ap.dummy_atom_idx]
            internal_atom = parent_3d.GetAtomWithIdx(ci["internal_idx"])
            parent_other = [n.GetIdx() for n in internal_atom.GetNeighbors()
                            if n.GetIdx() != ci["external_idx"]]
            if frag_other and parent_other:
                aligned_neighbor = aligned_tmp[ap.neighbor_atom_idx]
                frag_ref = np.mean([aligned_tmp[i] for i in frag_other], axis=0)
                frag_ref_dir = frag_ref - aligned_neighbor
                parent_int_pos = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))
                parent_ref = np.mean([np.array(list(parent_conf.GetAtomPosition(i)))
                                      for i in parent_other], axis=0)
                parent_ref_dir = parent_ref - parent_int_pos
                refine = refine_rotation_around_axis(
                    target_pos, axis, frag_ref_dir, parent_ref_dir,
                )
                transform = refine @ transform

    elif n_aps >= 2:
        source_pts = []
        target_pts = []
        for i in range(n_aps):
            ap = frag_aps[i]
            ci = cut_info[i]
            frag_neighbor_pos = np.array(list(conf.GetAtomPosition(ap.neighbor_atom_idx)))
            frag_dummy_pos = np.array(list(conf.GetAtomPosition(ap.dummy_atom_idx)))
            target_ext = np.array(list(parent_conf.GetAtomPosition(ci["external_idx"])))
            target_int = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))
            # neighbor → internal position, dummy → external position
            source_pts.append(frag_neighbor_pos)
            source_pts.append(frag_dummy_pos)
            target_pts.append(target_int)
            target_pts.append(target_ext)

        transform = align_two_vectors(np.array(source_pts), np.array(target_pts))
    else:
        return frag_mol  # no APs, can't align

    # Apply transform and create aligned mol
    aligned_positions = apply_transform(positions, transform)
    aligned_mol = Chem.RWMol(frag_mol)
    aligned_conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        aligned_conf.SetAtomPosition(
            i, Point3D(float(aligned_positions[i][0]),
                       float(aligned_positions[i][1]),
                       float(aligned_positions[i][2]))
        )

    # Replace existing conformers with aligned one
    aligned_mol.RemoveAllConformers()
    aligned_conf.SetId(0)
    aligned_mol.AddConformer(aligned_conf, assignId=True)
    return aligned_mol.GetMol()


def get_cut_info_from_parent(parent_3d, match_atoms):
    """Get cut bond info from the parent molecule (mirrors assemble.py logic)."""
    match_set = set(match_atoms)

    # Expand to include Hs exclusively bonded to matched atoms
    for atom in parent_3d.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIdx() not in match_set:
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1 and neighbors[0].GetIdx() in match_set:
                match_set.add(atom.GetIdx())

    cut_info = []
    for bond in parent_3d.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            external = a1 if a1 not in match_set else a2
            internal = a2 if a1 not in match_set else a1
            if parent_3d.GetAtomWithIdx(external).GetAtomicNum() == 1:
                continue
            cut_info.append({
                "external_idx": external,
                "internal_idx": internal,
            })
    return cut_info


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Inspect CAVEAT results as SDF files")
    parser.add_argument("--db", required=True, help="Path to fragment database")
    mol_group = parser.add_mutually_exclusive_group(required=True)
    mol_group.add_argument("--mol", help="Query molecule SMILES")
    mol_group.add_argument("--mol-sdf", help="Query molecule SDF file (preserves 3D coordinates)")
    parser.add_argument("--replace", required=True, help="SMARTS of substructure to replace")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--top", type=int, default=5, help="Number of replacements to query")
    parser.add_argument("--filter", dest="filter_str", default=None,
                        help="Absolute property filters, e.g. 'MW:100-700,cLogP:-2-6'. "
                             "Default: MW:100-750,cLogP:-3-7,HBA:0-12,HBD:0-6,RotBonds:0-15,TPSA:0-200")
    parser.add_argument("--delta-filter", dest="delta_filter_str", default=None,
                        help="Delta property filters (vs parent), e.g. 'MW<15,cLogP<0.5,HBA=0,ArRings=-1,TPSA=any'. "
                             "Use KEY<VAL for |delta|<VAL, KEY=VAL for exact delta, KEY=any to skip.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable all property filters")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Geometric tolerance multiplier (default 1.0 = d±0.5A, angles±15deg, "
                             "dihedral±30deg). Use 0.5 for strict, 2.0 for loose.")
    parser.add_argument("--optimize", action="store_true",
                        help="Run MMFF optimization on assembled products (default: off)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "fragments"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "assembled"), exist_ok=True)

    # Parse filters — delta-filter takes precedence over absolute filter
    if args.no_filter:
        filters = {}
        delta_filters = {}
    elif args.delta_filter_str:
        filters = {}  # no absolute filters when using delta
        delta_filters = parse_delta_filter_arg(args.delta_filter_str)
    elif args.filter_str:
        filters = parse_filter_arg(args.filter_str)
        delta_filters = {}
    else:
        filters = DEFAULT_FILTERS
        delta_filters = {}

    # --- 1. Parent molecule (reference coordinate frame) ---
    print("\n=== Parent Molecule ===")

    if args.mol_sdf:
        # Read from SDF — preserves docked/experimental 3D coordinates
        parent_3d = Chem.MolFromMolFile(args.mol_sdf, removeHs=True)
        if parent_3d is None:
            print(f"ERROR: Could not read SDF file: {args.mol_sdf}")
            sys.exit(1)
        mol_smiles = Chem.MolToSmiles(parent_3d)
        print(f"  Read from SDF: {args.mol_sdf}")
        print(f"  SMILES: {mol_smiles}")
        print(f"  Atoms: {parent_3d.GetNumAtoms()}, Conformers: {parent_3d.GetNumConformers()}")
        parent_for_match = parent_3d
    else:
        parent_2d = Chem.MolFromSmiles(args.mol)
        if parent_2d is None:
            print("ERROR: Could not parse molecule SMILES")
            sys.exit(1)
        mol_smiles = args.mol
        parent_3d = make_3d(parent_2d)
        parent_for_match = parent_2d

    # Compute parent properties
    parent_props = compute_properties(parent_for_match)
    print(f"  Properties: MW={parent_props['MW']}, cLogP={parent_props['cLogP']}, "
          f"HBA={parent_props['HBA']}, HBD={parent_props['HBD']}, "
          f"RotBonds={parent_props['RotBonds']}, TPSA={parent_props['TPSA']}, "
          f"ArRings={parent_props['ArRings']}, SatRings={parent_props['SatRings']}")

    write_sdf(parent_3d, os.path.join(args.outdir, "parent.sdf"),
              props={"SMILES": mol_smiles, "Name": "parent"})

    # Match substructure
    query_pat = Chem.MolFromSmarts(args.replace)
    matches = parent_for_match.GetSubstructMatches(query_pat)
    if not matches:
        print(f"ERROR: SMARTS '{args.replace}' not found in molecule")
        sys.exit(1)

    match_atoms = matches[0]
    print(f"  Matched atoms: {match_atoms}")

    parent_3d.SetProp("MatchedAtomIndices", ",".join(str(a) for a in match_atoms))
    write_sdf(parent_3d, os.path.join(args.outdir, "parent_highlight.sdf"),
              props={"SMILES": mol_smiles, "MatchedSMARTS": args.replace})

    # Match on 3D parent for assembly and alignment
    matches_3d = parent_3d.GetSubstructMatches(query_pat)
    if not matches_3d:
        print("  WARNING: Could not match SMARTS on 3D parent, skipping assembly")
        db = FragmentDatabase(args.db)
        db.close()
        return

    match_atoms_3d = matches_3d[0]
    cut_info = get_cut_info_from_parent(parent_3d, match_atoms_3d)

    # Build H-capped substructure for delta estimation
    sub_mol = build_hcapped_substructure(parent_for_match, match_atoms)
    sub_props = compute_properties(sub_mol) if sub_mol else None

    # --- 2. Query database ---
    print(f"\n=== Querying Database ({args.db}) ===")
    db = FragmentDatabase(args.db)

    # Find cut bonds on the matching mol for geometric descriptor computation
    cut_bonds = _find_cut_bonds(parent_for_match, set(match_atoms))
    num_aps = len(cut_bonds)

    if delta_filters and sub_props:
        # PROPERTY-FIRST search: scan ALL fragments by property delta,
        # then rank survivors by geometric distance. This avoids the
        # KDTree bottleneck where property-matching fragments are
        # geometrically distant and invisible to nearest-neighbor search.
        print(f"  Mode: property-first search (delta filters active)")
        print(f"  H-capped substructure: {Chem.MolToSmiles(sub_mol)}")
        print(f"  Cut bonds: {num_aps}")
        results = find_replacements_property_first(
            db, parent_3d, set(match_atoms_3d), cut_bonds, num_aps,
            sub_props, delta_filters, top_k=args.top,
        )
    else:
        # Standard geometry-first search with hard tolerance cutoffs
        tol = args.tolerance
        base_tol = {"distance": 0.5, "angle1": 15.0, "angle2": 15.0, "dihedral": 30.0}
        actual_tol = {k: v * tol for k, v in base_tol.items()}
        print(f"  Tolerances (x{tol}): d±{actual_tol['distance']:.2f}A, "
              f"angles±{actual_tol['angle1']:.0f}deg, dihedral±{actual_tol['dihedral']:.0f}deg")
        mol_3d_arg = parent_3d if args.mol_sdf else None
        results = find_replacements(
            parent_for_match, args.replace, db, n_confs=5, top_k=args.top,
            tolerance=tol, use_kdtree=False,
            mol_3d=mol_3d_arg,
        )

    print(f"  Found {len(results)} candidate replacement(s)")

    print(f"\n=== Assembling & Computing Properties ({len(results)} candidates) ===")
    all_products = []  # list of (rank, result, assembled_mol, frag_info, props, pass_filter)

    for i, result in enumerate(results, 1):
        frag_info = db.get_fragment(result.fragment_id)
        if frag_info is None or frag_info["mol"] is None:
            print(f"  {i:3d}. Fragment {result.fragment_id}: not available, skipping")
            continue

        frag_mol = frag_info["mol"]
        # Always detect APs dynamically from the actual mol to avoid
        # index mismatches between attachment_points_json and mol_binary
        ap_objects = _find_attachment_points_in_mol(frag_mol)

        assembled = assemble(parent_3d, match_atoms_3d, frag_mol, ap_objects,
                             optimize=args.optimize)
        if assembled is None:
            print(f"  {i:3d}. Assembly failed: {result.smiles}")
            continue

        props = compute_properties(assembled)
        passed = passes_filters(props, filters) and passes_delta_filters(props, parent_props, delta_filters)
        all_products.append((i, result, assembled, frag_info, props, passed))

    n_passed = sum(1 for p in all_products if p[5])
    n_failed = sum(1 for p in all_products if not p[5])
    print(f"  Assembled: {len(all_products)}, Passed filters: {n_passed}, Filtered out: {n_failed}")

    if filters:
        print(f"  Absolute filters: {', '.join(f'{k}:{lo}-{hi}' for k, (lo, hi) in filters.items())}")
    if delta_filters:
        parts = []
        for k, (lo, hi) in delta_filters.items():
            if lo == hi:
                parts.append(f"d{k}={lo:g}")
            else:
                parts.append(f"|d{k}|<{hi:g}")
        print(f"  Delta filters: {', '.join(parts)}")

    # --- 4. Write results (only products that pass filters) ---
    passed_products = [p for p in all_products if p[5]]

    print(f"\n=== Writing Fragments (passed filter) ===")
    for rank_out, (rank_in, result, assembled, frag_info, props, _) in enumerate(passed_products, 1):
        frag_mol = frag_info["mol"]
        if frag_mol is None or frag_mol.GetNumConformers() == 0:
            continue

        aps = _find_attachment_points_in_mol(frag_mol)

        aligned_frag = align_fragment_to_parent(frag_mol, aps, parent_3d, cut_info)

        path = os.path.join(args.outdir, "fragments", f"frag_{rank_out:03d}.sdf")
        write_sdf(aligned_frag, path, props={
            "Name": f"fragment_{result.fragment_id}",
            "SMILES": result.smiles,
            "GeometricDistance": f"{result.geometric_distance:.4f}",
            "NumAttachmentPoints": str(result.num_attachment_points),
            "NumHeavyAtoms": str(result.num_heavy_atoms),
            "SourceCount": str(result.source_count),
            "Rank": str(rank_out),
        })

    print(f"\n=== Writing Assembled Products (passed filter) ===")
    for rank_out, (rank_in, result, assembled, frag_info, props, _) in enumerate(passed_products, 1):
        assembled_smi = Chem.MolToSmiles(assembled)
        path = os.path.join(args.outdir, "assembled", f"assembled_{rank_out:03d}.sdf")

        # Store properties on the SDF
        prop_dict = {
            "Name": f"assembled_{rank_out}",
            "SMILES": assembled_smi,
            "ReplacementSMILES": result.smiles,
            "ReplacementFragmentID": str(result.fragment_id),
            "GeometricDistance": f"{result.geometric_distance:.4f}",
            "Rank": str(rank_out),
        }
        for k, v in props.items():
            prop_dict[k] = str(v)
        # Delta properties vs parent
        for k in ["MW", "cLogP", "HBA", "HBD", "RotBonds", "TPSA", "ArRings", "SatRings", "HeavyAtoms"]:
            delta = props[k] - parent_props[k]
            prop_dict[f"delta_{k}"] = f"{delta:+.1f}" if isinstance(props[k], float) else f"{delta:+d}"

        write_sdf(assembled, path, props=prop_dict)

    # --- 5. Combined multi-mol SDF files ---
    print(f"\n=== Combined Files ===")

    frag_writer = Chem.SDWriter(os.path.join(args.outdir, "all_fragments.sdf"))
    for rank_out, (rank_in, result, assembled, frag_info, props, _) in enumerate(passed_products, 1):
        frag_mol = frag_info["mol"]
        if frag_mol and frag_mol.GetNumConformers() > 0:
            aps = _find_attachment_points_in_mol(frag_mol)
            aligned_frag = align_fragment_to_parent(frag_mol, aps, parent_3d, cut_info)
            aligned_frag.SetProp("Name", f"frag_{result.fragment_id}")
            aligned_frag.SetProp("SMILES", result.smiles)
            aligned_frag.SetProp("Rank", str(rank_out))
            aligned_frag.SetProp("GeometricDistance", f"{result.geometric_distance:.4f}")
            frag_writer.write(aligned_frag)
    frag_writer.close()
    print(f"  Wrote: {os.path.join(args.outdir, 'all_fragments.sdf')}")

    asm_writer = Chem.SDWriter(os.path.join(args.outdir, "all_assembled.sdf"))
    for rank_out, (rank_in, result, assembled, frag_info, props, _) in enumerate(passed_products, 1):
        assembled.SetProp("Name", f"assembled_{rank_out}")
        assembled.SetProp("SMILES", Chem.MolToSmiles(assembled))
        assembled.SetProp("ReplacementSMILES", result.smiles)
        assembled.SetProp("Rank", str(rank_out))
        assembled.SetProp("GeometricDistance", f"{result.geometric_distance:.4f}")
        for k, v in props.items():
            assembled.SetProp(k, str(v))
        asm_writer.write(assembled)
    asm_writer.close()
    print(f"  Wrote: {os.path.join(args.outdir, 'all_assembled.sdf')}")

    # --- 6. Properties CSV ---
    csv_path = os.path.join(args.outdir, "properties.csv")
    prop_keys = ["MW", "cLogP", "HBA", "HBD", "RotBonds", "TPSA", "ArRings", "SatRings", "HeavyAtoms", "FormalCharge"]
    delta_keys = [f"d{k}" for k in prop_keys[:-1]]  # no delta for FormalCharge

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rank", "fragment_smiles", "assembled_smiles", "geo_score", "passed"] + prop_keys + delta_keys
        writer.writerow(header)

        # Write parent as row 0
        parent_row = [0, "(parent)", mol_smiles, "", ""]
        parent_row += [parent_props[k] for k in prop_keys]
        parent_row += ["" for _ in delta_keys]
        writer.writerow(parent_row)

        # Write all products (both passed and failed)
        for rank_in, result, assembled, frag_info, props, passed in all_products:
            assembled_smi = Chem.MolToSmiles(assembled)
            row = [rank_in, result.smiles, assembled_smi,
                   f"{result.geometric_distance:.4f}", "PASS" if passed else "FAIL"]
            row += [props[k] for k in prop_keys]
            row += [round(props[k] - parent_props[k], 2) for k in prop_keys[:-1]]
            writer.writerow(row)

    print(f"  Wrote: {csv_path}")

    # --- 7. Summary table ---
    show_delta = bool(delta_filters)
    print(f"\n=== Property Summary {'(deltas)' if show_delta else ''} ===")

    if show_delta:
        hdr = f"{'#':<4}{'Pass':<6}{'Score':<8}{'dMW':<8}{'dcLogP':<8}{'dHBA':<6}{'dHBD':<6}{'dRotB':<7}{'dTPSA':<8}{'dArR':<6}{'dSatR':<7}{'Fragment SMILES'}"
        print(hdr)
        print("-" * len(hdr))

        pp = parent_props
        print(f"{'P':<4}{'---':<6}{'':<8}{'0':<8}{'0':<8}{'0':<6}{'0':<6}{'0':<7}{'0':<8}{'0':<6}{'0':<7}(parent)")

        for rank_in, result, assembled, frag_info, props, passed in all_products:
            flag = "OK" if passed else "FAIL"
            smi = result.smiles
            if len(smi) > 45:
                smi = smi[:42] + "..."
            d = {k: props[k] - pp[k] for k in props if k in pp}
            print(f"{rank_in:<4}{flag:<6}{result.geometric_distance:<8.3f}"
                  f"{d['MW']:+<8.1f}{d['cLogP']:+<8.2f}{d['HBA']:+<6d}{d['HBD']:+<6d}"
                  f"{d['RotBonds']:+<7d}{d['TPSA']:+<8.1f}{d['ArRings']:+<6d}{d['SatRings']:+<7d}{smi}")
    else:
        hdr = f"{'#':<4}{'Pass':<6}{'Score':<8}{'MW':<8}{'cLogP':<8}{'HBA':<5}{'HBD':<5}{'RotB':<6}{'TPSA':<8}{'ArR':<5}{'SatR':<6}{'Fragment SMILES'}"
        print(hdr)
        print("-" * len(hdr))

        pp = parent_props
        print(f"{'P':<4}{'---':<6}{'':<8}{pp['MW']:<8}{pp['cLogP']:<8}{pp['HBA']:<5}{pp['HBD']:<5}"
              f"{pp['RotBonds']:<6}{pp['TPSA']:<8}{pp['ArRings']:<5}{pp['SatRings']:<6}(parent)")

        for rank_in, result, assembled, frag_info, props, passed in all_products:
            flag = "OK" if passed else "FAIL"
            smi = result.smiles
            if len(smi) > 45:
                smi = smi[:42] + "..."
            print(f"{rank_in:<4}{flag:<6}{result.geometric_distance:<8.3f}"
                  f"{props['MW']:<8}{props['cLogP']:<8}{props['HBA']:<5}{props['HBD']:<5}"
                  f"{props['RotBonds']:<6}{props['TPSA']:<8}{props['ArRings']:<5}{props['SatRings']:<6}{smi}")

    db.close()
    print(f"\n=== Done! {n_passed} products passed filters, {n_failed} filtered out ===")
    print(f"All structures are in the same coordinate frame — overlay should be precise.")


if __name__ == "__main__":
    main()
