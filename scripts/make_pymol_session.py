"""Build a PyMOL .pml script with assembled products and their source approved drugs.

For each hit fragment, traces back to the source approved drug, generates a 3D
conformer, aligns the drug so the fragment portion overlaps the placed fragment,
and writes everything out with a .pml loader script.

Usage:
    python scripts/make_pymol_session.py \
        --results-dir results_tyrout_approved2 \
        --frags-db approved_drugs_frags.db \
        --screen-db approved_drugs_screen.db
"""

import argparse
import csv
import os
import sqlite3
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors
from rdkit.Geometry import Point3D


def strip_dummies(mol):
    """Remove dummy atoms (*) from a mol, return (edited mol, mapping old->new idx)."""
    rw = Chem.RWMol(mol)
    dummy_idxs = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,
    )
    for idx in dummy_idxs:
        rw.RemoveAtom(idx)
    return rw.GetMol()


def get_fragment_core_smiles(frag_smi):
    """Get canonical SMILES of fragment with dummies removed."""
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        return None
    core = strip_dummies(mol)
    return Chem.MolToSmiles(core)


def align_drug_to_fragment(drug_mol, frag_mol_aligned):
    """Align a source drug 3D mol so its fragment portion overlaps the aligned fragment.

    frag_mol_aligned: fragment mol with 3D coords in the parent frame (from SDF).
    drug_mol: source drug mol with a 3D conformer.

    Returns aligned drug mol, or None on failure.
    """
    # Get the fragment core (no dummies) for substructure matching
    frag_core = strip_dummies(frag_mol_aligned)
    frag_core_aligned = strip_dummies(frag_mol_aligned)

    # Build atom map: which fragment atoms (non-dummy) correspond to which indices
    # in the original fragment mol
    frag_nondummy = [
        a.GetIdx() for a in frag_mol_aligned.GetAtoms() if a.GetAtomicNum() != 0
    ]

    # Match core in the drug
    match_in_drug = drug_mol.GetSubstructMatch(frag_core)
    if not match_in_drug:
        # Try with generic query
        frag_core_q = Chem.MolFromSmarts(Chem.MolToSmarts(frag_core))
        if frag_core_q:
            match_in_drug = drug_mol.GetSubstructMatch(frag_core_q)
    if not match_in_drug:
        return None

    # Get coordinates for alignment
    frag_conf = frag_mol_aligned.GetConformer(0)
    drug_conf = drug_mol.GetConformer(0)

    # Build point pairs: frag non-dummy positions -> drug matched positions
    frag_pts = []
    drug_pts = []
    for i, frag_idx in enumerate(frag_nondummy):
        frag_pos = frag_conf.GetAtomPosition(frag_idx)
        drug_idx = match_in_drug[i]
        drug_pos = drug_conf.GetAtomPosition(drug_idx)
        frag_pts.append([frag_pos.x, frag_pos.y, frag_pos.z])
        drug_pts.append([drug_pos.x, drug_pos.y, drug_pos.z])

    frag_pts = np.array(frag_pts)
    drug_pts = np.array(drug_pts)

    if len(frag_pts) < 3:
        return None

    # Kabsch alignment: find R, t such that R @ drug_pts.T + t ≈ frag_pts.T
    frag_center = frag_pts.mean(axis=0)
    drug_center = drug_pts.mean(axis=0)

    P = frag_pts - frag_center
    Q = drug_pts - drug_center

    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    t = frag_center - R @ drug_center

    # Apply transform to all drug atoms
    aligned_drug = Chem.RWMol(drug_mol)
    conf = aligned_drug.GetConformer(0)
    for i in range(aligned_drug.GetNumAtoms()):
        pos = np.array([conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z])
        new_pos = R @ pos + t
        conf.SetAtomPosition(i, Point3D(*new_pos.tolist()))

    return aligned_drug.GetMol()


def lookup_source_drugs(frag_smiles, frags_db_path):
    """Look up source drug SMILES for a fragment SMILES.

    Tries exact match first, then strips isotope labels and tries again.
    """
    db = sqlite3.connect(frags_db_path)

    # Try exact match
    rows = db.execute(
        """SELECT DISTINCT s.source_smiles
           FROM sources s JOIN fragments f ON s.fragment_id = f.id
           WHERE f.canonical_smiles = ?
           LIMIT 5""",
        (frag_smiles,),
    ).fetchall()

    if not rows:
        # Try with all possible isotope label combinations stripped/re-applied
        # Get all 2-AP fragments from DB that match structurally
        mol = Chem.MolFromSmiles(frag_smiles)
        if mol is not None:
            # Strip isotopes and compare
            mol_no_iso = Chem.RWMol(mol)
            for atom in mol_no_iso.GetAtoms():
                atom.SetIsotope(0)
            bare_smi = Chem.MolToSmiles(mol_no_iso)

            # Get all fragments and check bare match
            all_frags = db.execute(
                "SELECT id, canonical_smiles FROM fragments WHERE num_attachment_points = ?",
                (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0),),
            ).fetchall()
            for fid, db_smi in all_frags:
                db_mol = Chem.MolFromSmiles(db_smi)
                if db_mol is None:
                    continue
                db_mol_no_iso = Chem.RWMol(db_mol)
                for atom in db_mol_no_iso.GetAtoms():
                    atom.SetIsotope(0)
                if Chem.MolToSmiles(db_mol_no_iso) == bare_smi:
                    src = db.execute(
                        "SELECT DISTINCT source_smiles FROM sources WHERE fragment_id = ? LIMIT 5",
                        (fid,),
                    ).fetchall()
                    if src:
                        rows = src
                        break

    db.close()
    return [r[0] for r in rows]


def lookup_drug_name(drug_smiles, smi_file="data/chembl_approved_drugs.smi"):
    """Look up the ChEMBL ID for a drug SMILES from the .smi file."""
    if not os.path.exists(smi_file):
        return None
    with open(smi_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == drug_smiles:
                return parts[1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Build PyMOL session from CAVEAT results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--frags-db", required=True)
    parser.add_argument("--screen-db", required=True)
    args = parser.parse_args()

    results_dir = args.results_dir
    drugs_dir = os.path.join(results_dir, "source_drugs")
    os.makedirs(drugs_dir, exist_ok=True)

    # Read properties.csv to get fragment SMILES per rank
    props_path = os.path.join(results_dir, "properties.csv")
    hits = []
    with open(props_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["rank"] == "0":
                continue  # parent row
            hits.append({
                "rank": int(row["rank"]),
                "frag_smi": row["fragment_smiles"],
                "assembled_smi": row["assembled_smiles"],
                "geo_score": row["geo_score"],
            })

    # For each fragment, look up the canonical SMILES in the screen DB
    screen_db = sqlite3.connect(args.screen_db)

    print(f"Processing {len(hits)} hits...")

    drug_info = []  # (rank, drug_smi, drug_name, aligned_sdf_path)

    for hit in hits:
        rank = hit["rank"]
        frag_smi = hit["frag_smi"]

        # The fragment SMILES in properties.csv has isotope labels like [5*]
        # which match the DB format. Just re-canonicalize to be safe.
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            print(f"  Rank {rank}: Cannot parse fragment SMILES {frag_smi}")
            drug_info.append((rank, None, None, None))
            continue

        canon_smi = Chem.MolToSmiles(frag_mol)

        # Look up source drugs
        source_drugs = lookup_source_drugs(canon_smi, args.frags_db)
        if not source_drugs:
            print(f"  Rank {rank}: No source drug found for {canon_smi}")
            drug_info.append((rank, None, None, None))
            continue

        drug_smi = source_drugs[0]
        drug_name = lookup_drug_name(drug_smi)

        # Load the aligned fragment SDF
        frag_sdf_path = os.path.join(results_dir, "fragments", f"frag_{rank:03d}.sdf")
        if not os.path.exists(frag_sdf_path):
            print(f"  Rank {rank}: No fragment SDF")
            drug_info.append((rank, drug_smi, drug_name, None))
            continue

        frag_mol_aligned = Chem.MolFromMolFile(frag_sdf_path, removeHs=True)
        if frag_mol_aligned is None:
            print(f"  Rank {rank}: Cannot read fragment SDF")
            drug_info.append((rank, drug_smi, drug_name, None))
            continue

        # Generate 3D for source drug
        drug_mol = Chem.MolFromSmiles(drug_smi)
        if drug_mol is None:
            print(f"  Rank {rank}: Cannot parse drug SMILES")
            drug_info.append((rank, drug_smi, drug_name, None))
            continue

        drug_mol = Chem.AddHs(drug_mol)
        if AllChem.EmbedMolecule(drug_mol, AllChem.ETKDGv3()) < 0:
            if AllChem.EmbedMolecule(drug_mol, AllChem.ETKDGv3()) < 0:
                print(f"  Rank {rank}: Drug embedding failed")
                drug_info.append((rank, drug_smi, drug_name, None))
                continue
        AllChem.MMFFOptimizeMolecule(drug_mol)
        drug_mol = Chem.RemoveHs(drug_mol)

        # Align drug to fragment
        aligned_drug = align_drug_to_fragment(drug_mol, frag_mol_aligned)
        if aligned_drug is None:
            print(f"  Rank {rank}: Drug alignment failed for {drug_name or drug_smi[:40]}")
            drug_info.append((rank, drug_smi, drug_name, None))
            continue

        # Write aligned drug SDF
        drug_sdf_path = os.path.join(drugs_dir, f"drug_{rank:03d}.sdf")
        writer = Chem.SDWriter(drug_sdf_path)
        aligned_drug.SetProp("_Name", f"drug_{rank:03d}")
        aligned_drug.SetProp("source_smiles", drug_smi)
        if drug_name:
            aligned_drug.SetProp("chembl_id", drug_name)
        aligned_drug.SetProp("fragment_smiles", frag_smi)
        writer.write(aligned_drug)
        writer.close()

        label = drug_name or drug_smi[:50]
        print(f"  Rank {rank}: {label} -> {drug_sdf_path}")
        drug_info.append((rank, drug_smi, drug_name, drug_sdf_path))

    screen_db.close()

    # Write PyMOL .pml script
    pml_path = os.path.join(results_dir, "session.pml")
    with open(pml_path, "w") as f:
        f.write("# CAVEAT results PyMOL session\n")
        f.write("# Load parent, then pairs of (assembled product, source drug)\n\n")
        f.write("bg_color white\n")
        f.write("set stick_radius, 0.15\n")
        f.write("set sphere_scale, 0.2\n\n")

        # Load parent
        f.write(f"load {os.path.abspath(os.path.join(results_dir, 'parent.sdf'))}, parent\n")
        f.write("show sticks, parent\n")
        f.write("color gray60, parent\n")
        f.write("disable parent\n\n")

        for rank, drug_smi, drug_name, drug_sdf_path in drug_info:
            assembled_path = os.path.abspath(
                os.path.join(results_dir, "assembled", f"assembled_{rank:03d}.sdf")
            )
            frag_path = os.path.abspath(
                os.path.join(results_dir, "fragments", f"frag_{rank:03d}.sdf")
            )

            f.write(f"# --- Rank {rank} ---\n")

            # Load assembled product
            f.write(f"load {assembled_path}, assembled_{rank:03d}\n")
            f.write(f"show sticks, assembled_{rank:03d}\n")
            f.write(f"color cyan, assembled_{rank:03d}\n")
            f.write(f"disable assembled_{rank:03d}\n")

            # Load source drug (aligned)
            if drug_sdf_path:
                drug_abs = os.path.abspath(drug_sdf_path)
                label = drug_name or f"drug_{rank:03d}"
                f.write(f"load {drug_abs}, {label}\n")
                f.write(f"show sticks, {label}\n")
                f.write(f"color salmon, {label}\n")
                f.write(f"disable {label}\n")

            f.write("\n")

        # Enable parent as reference
        f.write("enable parent\n")
        f.write("zoom parent\n")
        f.write("set_view auto\n")

    print(f"\nWrote PyMOL script: {pml_path}")
    print(f"Open in PyMOL: pymol {pml_path}")
    print(f"\nIn PyMOL, click through objects in the right panel.")
    print(f"Each assembled_NNN is paired with its source drug below it.")


if __name__ == "__main__":
    main()
