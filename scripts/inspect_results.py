#!/usr/bin/env python
"""Inspect CAVEAT pipeline results by writing SDF files.

Usage:
    python scripts/inspect_results.py \
        --db test.db \
        --mol "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5" \
        --replace "CN1CCNCC1" \
        --outdir results/ \
        --top 5

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

All structures are in the same coordinate frame as the parent molecule.
"""

import argparse
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D

from caveat.database import FragmentDatabase
from caveat.fragment import AttachmentPoint
from caveat.query import find_replacements, _find_cut_bonds
from caveat.assemble import assemble
from caveat.geometry import align_single_vector, align_two_vectors, apply_transform


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
        target_pos = np.array(list(parent_conf.GetAtomPosition(ci["external_idx"])))
        target_dir_pos = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))

        transform = align_single_vector(
            positions, frag_neighbor_pos, frag_dummy_pos,
            target_pos, target_dir_pos,
        )
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
            source_pts.append(frag_neighbor_pos)
            source_pts.append(frag_dummy_pos)
            target_pts.append(target_ext)
            target_pts.append(target_int)

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


def main():
    parser = argparse.ArgumentParser(description="Inspect CAVEAT results as SDF files")
    parser.add_argument("--db", required=True, help="Path to fragment database")
    parser.add_argument("--mol", required=True, help="Query molecule SMILES")
    parser.add_argument("--replace", required=True, help="SMARTS of substructure to replace")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--top", type=int, default=5, help="Number of replacements")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "fragments"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "assembled"), exist_ok=True)

    # --- 1. Parent molecule (reference coordinate frame) ---
    print("\n=== Parent Molecule ===")
    parent_2d = Chem.MolFromSmiles(args.mol)
    if parent_2d is None:
        print("ERROR: Could not parse molecule SMILES")
        sys.exit(1)

    parent_3d = make_3d(parent_2d)
    write_sdf(parent_3d, os.path.join(args.outdir, "parent.sdf"),
              props={"SMILES": args.mol, "Name": "parent"})

    # Match on 2D for display
    query_pat = Chem.MolFromSmarts(args.replace)
    matches = parent_2d.GetSubstructMatches(query_pat)
    if not matches:
        print(f"ERROR: SMARTS '{args.replace}' not found in molecule")
        sys.exit(1)

    match_atoms = matches[0]
    print(f"  Matched atoms (2D): {match_atoms}")

    parent_3d.SetProp("MatchedAtomIndices", ",".join(str(a) for a in match_atoms))
    write_sdf(parent_3d, os.path.join(args.outdir, "parent_highlight.sdf"),
              props={"SMILES": args.mol, "MatchedSMARTS": args.replace})

    # --- 2. Query database ---
    print(f"\n=== Querying Database ({args.db}) ===")
    db = FragmentDatabase(args.db)
    results = find_replacements(parent_2d, args.replace, db, n_confs=5, top_k=args.top)
    print(f"  Found {len(results)} replacement(s)")

    # Match on 3D parent for assembly and alignment
    matches_3d = parent_3d.GetSubstructMatches(query_pat)
    if not matches_3d:
        print("  WARNING: Could not match SMARTS on 3D parent, skipping assembly")
        db.close()
        return

    match_atoms_3d = matches_3d[0]
    cut_info = get_cut_info_from_parent(parent_3d, match_atoms_3d)

    # --- 3. Write fragment SDFs (aligned to parent frame) ---
    print(f"\n=== Replacement Fragments ===")
    for i, result in enumerate(results, 1):
        frag_info = db.get_fragment(result.fragment_id)
        if frag_info is None:
            print(f"  Fragment {result.fragment_id}: not found in DB")
            continue

        frag_mol = frag_info["mol"]
        if frag_mol is None or frag_mol.GetNumConformers() == 0:
            print(f"  Fragment {result.fragment_id}: no 3D conformer in DB")
            continue

        aps = frag_info["attachment_points"]
        if isinstance(aps[0], dict):
            aps = [AttachmentPoint.from_dict(a) for a in aps]

        # Align stored DB conformer to parent reference frame
        aligned_frag = align_fragment_to_parent(frag_mol, aps, parent_3d, cut_info)

        path = os.path.join(args.outdir, "fragments", f"frag_{i:03d}.sdf")
        write_sdf(aligned_frag, path, props={
            "Name": f"fragment_{result.fragment_id}",
            "SMILES": result.smiles,
            "GeometricDistance": f"{result.geometric_distance:.4f}",
            "NumAttachmentPoints": str(result.num_attachment_points),
            "NumHeavyAtoms": str(result.num_heavy_atoms),
            "SourceCount": str(result.source_count),
            "Rank": str(i),
        })

    # --- 4. Assemble and write product SDFs (automatically aligned via _place_coordinates) ---
    print(f"\n=== Assembled Products ===")
    for i, result in enumerate(results, 1):
        frag_info = db.get_fragment(result.fragment_id)
        if frag_info is None:
            continue

        frag_mol = frag_info["mol"]
        if frag_mol is None:
            continue

        aps = frag_info["attachment_points"]
        ap_objects = []
        for ap in aps:
            if isinstance(ap, dict):
                ap_objects.append(AttachmentPoint.from_dict(ap))
            else:
                ap_objects.append(ap)

        assembled = assemble(parent_3d, match_atoms_3d, frag_mol, ap_objects)
        if assembled is None:
            print(f"  Product {i}: assembly failed for fragment {result.fragment_id} ({result.smiles})")
            continue

        assembled_smi = Chem.MolToSmiles(assembled)
        path = os.path.join(args.outdir, "assembled", f"assembled_{i:03d}.sdf")
        write_sdf(assembled, path, props={
            "Name": f"assembled_{i}",
            "SMILES": assembled_smi,
            "ReplacementSMILES": result.smiles,
            "ReplacementFragmentID": str(result.fragment_id),
            "GeometricDistance": f"{result.geometric_distance:.4f}",
            "Rank": str(i),
        })

    # --- 5. Combined multi-mol SDF files ---
    print(f"\n=== Combined Files ===")

    # All fragments in one file (aligned to parent)
    frag_writer = Chem.SDWriter(os.path.join(args.outdir, "all_fragments.sdf"))
    for i, result in enumerate(results, 1):
        frag_info = db.get_fragment(result.fragment_id)
        if frag_info and frag_info["mol"] and frag_info["mol"].GetNumConformers() > 0:
            aps = frag_info["attachment_points"]
            if isinstance(aps[0], dict):
                aps = [AttachmentPoint.from_dict(a) for a in aps]
            aligned_frag = align_fragment_to_parent(frag_info["mol"], aps, parent_3d, cut_info)
            aligned_frag.SetProp("Name", f"frag_{result.fragment_id}")
            aligned_frag.SetProp("SMILES", result.smiles)
            aligned_frag.SetProp("Rank", str(i))
            aligned_frag.SetProp("GeometricDistance", f"{result.geometric_distance:.4f}")
            frag_writer.write(aligned_frag)
    frag_writer.close()
    print(f"  Wrote: {os.path.join(args.outdir, 'all_fragments.sdf')}")

    # All assembled in one file (already aligned via _place_coordinates)
    asm_writer = Chem.SDWriter(os.path.join(args.outdir, "all_assembled.sdf"))
    for i, result in enumerate(results, 1):
        frag_info = db.get_fragment(result.fragment_id)
        if frag_info and frag_info["mol"]:
            ap_objects = []
            for ap in frag_info["attachment_points"]:
                if isinstance(ap, dict):
                    ap_objects.append(AttachmentPoint.from_dict(ap))
                else:
                    ap_objects.append(ap)
            assembled = assemble(parent_3d, match_atoms_3d, frag_info["mol"], ap_objects)
            if assembled:
                assembled.SetProp("Name", f"assembled_{i}")
                assembled.SetProp("SMILES", Chem.MolToSmiles(assembled))
                assembled.SetProp("ReplacementSMILES", result.smiles)
                assembled.SetProp("Rank", str(i))
                asm_writer.write(assembled)
    asm_writer.close()
    print(f"  Wrote: {os.path.join(args.outdir, 'all_assembled.sdf')}")

    db.close()
    print(f"\n=== Done! Open SDFs in PyMOL, Maestro, or any molecular viewer ===")
    print(f"All structures are in the same coordinate frame — overlay should be precise.")


if __name__ == "__main__":
    main()
