#!/usr/bin/env python
"""Second-order fragment screening: extend fragments with small linkers.

For a given query (parent + substructure to replace), finds additional replacement
candidates by inserting small linker atoms (CH2, NH, O, etc.) between existing
database fragments and their attachment points.

This enables finding replacements that:
1. Span longer distances than any single database fragment
2. Create novel chemistry at junctions (e.g., adding a methylene spacer)
3. Mix and match fragment cores with linker atoms

Steps:
1. Compute query descriptor from parent molecule
2. Query DB fragments and identify those that are "too short"
3. Extend candidates with linker library
4. Embed extended fragments, compute descriptors, score vs query
5. Assemble into parent and evaluate (properties + OOP)
6. Report novel hits vs. direct screening

Usage:
    python scripts/second_order_screen.py
"""
import csv
import json
import os
import re
import sqlite3
import sys
import time
import multiprocessing as mp
import numpy as np
from itertools import combinations

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

from caveat.geometry import embed_fragment, compute_vector_pair_descriptor
from caveat.database import FragmentDatabase, _find_attachment_points_in_mol
from caveat.query import find_replacements, _strip_brics_labels
from caveat.assemble import assemble, compute_planarity_score

RDLogger.DisableLog("rdApp.*")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ─── Configuration ───────────────────────────────────────────────────────
DB_PATH = "/Users/rafal/repos/CAVEAT/chembl_screen_rot1_5conf.db"
PARENT_SDF = "/Users/rafal/repos/CAVEAT/results_tyrout_approved/parent.sdf"
REPLACE_SMI = "COc1cnnc1"
OUTDIR = "/Users/rafal/repos/CAVEAT/results_tyrout_2nd_order_v2"
NUM_WORKERS = 14
N_CONFS = 5
TOP_N = 99999  # assemble all hits
GEO_SCORE_CUTOFF = 2.0

# Fragment property filters (applied BEFORE embedding)
FRAG_MAX_HBD = 0
FRAG_MAX_HBA = 2

# ─── Linker Library ──────────────────────────────────────────────────────
# (name, atomic_numbers to insert as single bonds)
SIMPLE_LINKERS = [
    ("CH2", [6]),
    ("NH", [7]),
    ("O", [8]),
    ("CH2CH2", [6, 6]),
    ("CH2O", [6, 8]),
    ("OCH2", [8, 6]),
    ("CH2NH", [6, 7]),
    ("NHCH2", [7, 6]),
]


def extend_at_ap(frag_mol, ap_idx, linker_atoms):
    """Insert linker atoms between a dummy atom and its neighbor.

    The dummy atom is preserved (just moved farther from the fragment core).
    AP count stays the same.

    Args:
        frag_mol: RDKit Mol with dummy atoms ([n*])
        ap_idx: which AP to extend (0-indexed among dummy atoms)
        linker_atoms: list of atomic numbers, e.g. [6] for CH2

    Returns:
        Extended Mol or None if sanitization fails.
    """
    dummies = []
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummies.append(atom.GetIdx())
    if ap_idx >= len(dummies):
        return None

    dummy_idx = dummies[ap_idx]
    dummy_atom = frag_mol.GetAtomWithIdx(dummy_idx)
    neighbors = dummy_atom.GetNeighbors()
    if len(neighbors) != 1:
        return None
    neighbor_idx = neighbors[0].GetIdx()

    rw = Chem.RWMol(frag_mol)
    rw.RemoveBond(dummy_idx, neighbor_idx)

    prev_idx = neighbor_idx
    for anum in linker_atoms:
        new_idx = rw.AddAtom(Chem.Atom(anum))
        rw.AddBond(prev_idx, new_idx, Chem.BondType.SINGLE)
        prev_idx = new_idx

    rw.AddBond(prev_idx, dummy_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _embed_and_score(args):
    """Worker: embed an extended fragment, compute descriptor, score vs query.

    Returns best hit as dict or None.
    """
    ext_smi, orig_smi, linker_name, ap_side, query_norm = args

    mol = Chem.MolFromSmiles(ext_smi)
    if mol is None:
        return None

    try:
        mol3d = embed_fragment(mol, n_confs=N_CONFS)
    except Exception:
        return None

    if mol3d is None or mol3d.GetNumConformers() == 0:
        return None

    aps = _find_attachment_points_in_mol(mol3d)
    if len(aps) < 2:
        return None

    best_score = 999.0
    best_conf = 0
    best_desc = None

    for conf_idx in range(mol3d.GetNumConformers()):
        conf = mol3d.GetConformer(conf_idx)
        for (ai, ap1), (aj, ap2) in combinations(enumerate(aps), 2):
            desc = compute_vector_pair_descriptor(
                conf,
                ap1.dummy_atom_idx, ap1.neighbor_atom_idx,
                ap2.dummy_atom_idx, ap2.neighbor_atom_idx,
            )
            desc_norm = np.array([
                desc.distance / 1.0,
                desc.angle1 / 15.0,
                desc.angle2 / 15.0,
                desc.dihedral / 30.0,
            ])
            score = float(np.linalg.norm(desc_norm - query_norm))
            if score < best_score:
                best_score = score
                best_conf = conf_idx
                best_desc = desc

    if best_score >= GEO_SCORE_CUTOFF or best_desc is None:
        return None

    return {
        "ext_smi": ext_smi,
        "orig_smi": orig_smi,
        "linker": linker_name,
        "ap_side": ap_side,
        "score": best_score,
        "conf_idx": best_conf,
        "distance": best_desc.distance,
        "angle1": best_desc.angle1,
        "angle2": best_desc.angle2,
        "dihedral": best_desc.dihedral,
        "mol_binary": mol3d.ToBinary(),
    }


def main():
    t0 = time.time()

    # ─── 1. Load parent molecule ──────────────────────────────────
    print("=" * 70)
    print("SECOND-ORDER FRAGMENT SCREENING")
    print("=" * 70)
    print(f"\nLoading parent from {PARENT_SDF}...")
    suppl = Chem.SDMolSupplier(PARENT_SDF, removeHs=False)
    parent_3d = next(suppl)
    parent_noH = Chem.RemoveHs(parent_3d)

    query_pat = Chem.MolFromSmarts(REPLACE_SMI)
    match = parent_noH.GetSubstructMatch(query_pat)
    if not match:
        print(f"ERROR: '{REPLACE_SMI}' not found in parent")
        return
    match_atoms = tuple(match)
    match_set = set(match)
    print(f"  Match atoms: {match_atoms}")

    # ─── 2. Compute query descriptor ─────────────────────────────
    conf = parent_noH.GetConformer(0)
    cut_bonds = []
    for bond in parent_noH.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            internal = a1 if a1 in match_set else a2
            external = a2 if a1 in match_set else a1
            cut_bonds.append((internal, external))

    if len(cut_bonds) != 2:
        print(f"ERROR: Expected 2 cut bonds, got {len(cut_bonds)}")
        return

    (int1, ext1), (int2, ext2) = cut_bonds
    query_desc = compute_vector_pair_descriptor(conf, ext1, int1, ext2, int2)
    query_norm = np.array([
        query_desc.distance / 1.0,
        query_desc.angle1 / 15.0,
        query_desc.angle2 / 15.0,
        query_desc.dihedral / 30.0,
    ])
    print(f"  Query: D={query_desc.distance:.2f}A  "
          f"A1={query_desc.angle1:.1f}  A2={query_desc.angle2:.1f}  "
          f"Dih={query_desc.dihedral:.1f}")

    # ─── 3. Direct screening baseline ────────────────────────────
    print("\nDirect screening (baseline)...")
    db = FragmentDatabase(DB_PATH)
    # Use SQL range query (faster than KDTree for collecting all hits)
    direct_results = find_replacements(
        parent_noH, REPLACE_SMI, db, top_k=10000,
        tolerance=2.0, use_kdtree=False, mol_3d=parent_noH,
    )
    direct_cores = {_strip_brics_labels(r.smiles) for r in direct_results}
    direct_smiles = {r.smiles for r in direct_results}
    print(f"  Direct hits: {len(direct_results)} ({len(direct_cores)} unique cores)")

    # ─── 4. Find "too short" fragments ───────────────────────────
    print("\nAnalyzing fragment distances...")
    conn = sqlite3.connect(DB_PATH)
    frag_rows = conn.execute("""
        SELECT f.id, f.canonical_smiles, f.num_heavy_atoms,
               MAX(vp.distance) as max_dist
        FROM fragments f
        JOIN vector_pairs vp ON vp.fragment_id = f.id
        GROUP BY f.id
    """).fetchall()
    conn.close()

    qd = query_desc.distance
    short_frags = []
    for fid, smi, nha, max_d in frag_rows:
        shortfall = qd - max_d
        if 0.3 < shortfall < 6.0 and nha <= 12:
            short_frags.append((fid, smi, nha, max_d, shortfall))

    short_frags.sort(key=lambda x: x[4])  # sort by shortfall (smallest first)
    print(f"  Query distance: {qd:.2f} A")
    print(f"  Total 2-AP fragments: {len(frag_rows):,}")
    print(f"  'Too short' candidates (0.3-6.0 A, HA<=12): {len(short_frags):,}")

    # ─── 5. Generate extended fragments ──────────────────────────
    print(f"\nExtending with {len(SIMPLE_LINKERS)} linkers "
          f"(fragment filter: HBD<={FRAG_MAX_HBD}, HBA<={FRAG_MAX_HBA})...")
    tasks = []
    seen = set()
    n_invalid = 0
    n_prop_filtered = 0

    for fid, smi, nha, max_d, shortfall in short_frags:
        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol is None:
            continue

        for linker_name, linker_atoms in SIMPLE_LINKERS:
            if nha + len(linker_atoms) > 14:
                continue

            for ap_side in range(2):
                ext_mol = extend_at_ap(frag_mol, ap_side, linker_atoms)
                if ext_mol is None:
                    n_invalid += 1
                    continue

                # Fragment property filter (before expensive embedding)
                hbd = Descriptors.NumHDonors(ext_mol)
                hba = Descriptors.NumHAcceptors(ext_mol)
                if hbd > FRAG_MAX_HBD or hba > FRAG_MAX_HBA:
                    n_prop_filtered += 1
                    continue

                ext_smi = Chem.MolToSmiles(ext_mol)
                ext_core = _strip_brics_labels(ext_smi)

                # Skip if already in direct screening
                if ext_core in direct_cores:
                    continue
                if ext_smi in seen:
                    continue
                seen.add(ext_smi)

                tasks.append((ext_smi, smi, linker_name, ap_side, query_norm))

    print(f"  Valid extensions: {len(tasks):,} unique")
    print(f"  Skipped (invalid valence): {n_invalid:,}")
    print(f"  Skipped (HBD>{FRAG_MAX_HBD} or HBA>{FRAG_MAX_HBA}): {n_prop_filtered:,}")
    print(f"  Skipped (already in direct): {len(seen) - len(tasks) + len(direct_cores):,}")

    # ─── 6. Embed and score (parallel) ───────────────────────────
    print(f"\nEmbedding and scoring with {NUM_WORKERS} workers...")
    t1 = time.time()
    hits = []
    n_processed = 0

    pool = mp.Pool(NUM_WORKERS)
    for result in pool.imap_unordered(_embed_and_score, tasks, chunksize=50):
        n_processed += 1
        if result is not None:
            hits.append(result)
        if n_processed % 2000 == 0:
            elapsed = time.time() - t1
            rate = n_processed / elapsed
            eta = (len(tasks) - n_processed) / rate if rate > 0 else 0
            print(f"  {n_processed:>7,}/{len(tasks):,}  "
                  f"{len(hits)} hits  {rate:.0f}/s  ETA: {eta/60:.1f}m")

    pool.close()
    pool.join()
    elapsed = time.time() - t1
    print(f"  Done! {n_processed:,} processed in {elapsed:.0f}s, {len(hits)} hits")

    # ─── 7. Rank, deduplicate, and filter ────────────────────────
    hits.sort(key=lambda h: h["score"])

    # Deduplicate by core SMILES
    seen_cores = set()
    unique_hits = []
    for h in hits:
        core = _strip_brics_labels(h["ext_smi"])
        if core not in seen_cores and core not in direct_cores:
            seen_cores.add(core)
            unique_hits.append(h)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Direct screening:     {len(direct_results):>5} hits")
    print(f"  2nd-order screening:  {len(unique_hits):>5} novel hits")

    if not unique_hits:
        print("  No novel hits found!")
        db.close()
        return

    # Show top hits (sorted by score)
    top = unique_hits[:min(20, len(unique_hits))]
    print(f"\n  Top {len(top)} novel 2nd-order hits (by geo score):")
    print(f"  {'Rank':>4} {'Score':>6} {'D(A)':>5} {'Linker':>8} {'HA':>3} {'HBA':>3} {'Fragment'}")
    for i, h in enumerate(top):
        ext_mol = Chem.MolFromSmiles(h["ext_smi"])
        nha = ext_mol.GetNumHeavyAtoms() if ext_mol else "?"
        hba = Descriptors.NumHAcceptors(ext_mol) if ext_mol else "?"
        smi = h["ext_smi"]
        if len(smi) > 40:
            smi = smi[:37] + "..."
        print(f"  {i+1:>4} {h['score']:>6.3f} {h['distance']:>5.2f} "
              f"{h['linker']:>8} {nha:>3} {hba:>3} {smi}")

    # ─── 8. Assemble top hits ────────────────────────────────────
    # Sort by HBA (ascending), then by score within same HBA
    for h in unique_hits:
        ext_mol = Chem.MolFromSmiles(h["ext_smi"])
        h["frag_hba"] = Descriptors.NumHAcceptors(ext_mol) if ext_mol else 99
        h["frag_hbd"] = Descriptors.NumHDonors(ext_mol) if ext_mol else 99
    unique_hits.sort(key=lambda h: (h["frag_hba"], h["score"]))

    to_assemble = unique_hits[:TOP_N]
    print(f"\nAssembling {len(to_assemble)} hits (sorted by frag HBA, then score)...")

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(f"{OUTDIR}/assembled", exist_ok=True)
    os.makedirs(f"{OUTDIR}/fragments", exist_ok=True)

    # Get parent match atoms for 3D parent (with Hs)
    match_3d = parent_3d.GetSubstructMatch(query_pat)
    if not match_3d:
        # Try on noH version
        match_3d = match_atoms

    # Write parent
    writer = Chem.SDWriter(f"{OUTDIR}/parent.sdf")
    writer.write(parent_3d)
    writer.close()

    assembled_results = []
    all_assembled_writer = Chem.SDWriter(f"{OUTDIR}/all_assembled.sdf")
    all_frag_writer = Chem.SDWriter(f"{OUTDIR}/all_fragments.sdf")

    for rank, h in enumerate(to_assemble, 1):
        mol3d = Chem.Mol(h["mol_binary"])
        aps = _find_attachment_points_in_mol(mol3d)
        if len(aps) < 2:
            continue

        try:
            assembled = assemble(
                parent_3d, match_3d, mol3d, aps,
                replacement_conf_id=h["conf_idx"],
            )
        except Exception as e:
            assembled_results.append({
                "rank": rank, "ext_smi": h["ext_smi"], "orig_smi": h["orig_smi"],
                "linker": h["linker"], "score": h["score"],
                "status": f"FAIL: {e}",
            })
            continue

        if assembled is None:
            assembled_results.append({
                "rank": rank, "ext_smi": h["ext_smi"], "orig_smi": h["orig_smi"],
                "linker": h["linker"], "score": h["score"],
                "status": "FAIL: assemble returned None",
            })
            continue

        # Compute properties
        assembled_smi = Chem.MolToSmiles(Chem.RemoveHs(assembled))
        props = {
            "MW": Descriptors.MolWt(assembled),
            "cLogP": Descriptors.MolLogP(assembled),
            "HBA": Descriptors.NumHAcceptors(assembled),
            "HBD": Descriptors.NumHDonors(assembled),
            "RotBonds": Descriptors.NumRotatableBonds(assembled),
            "TPSA": Descriptors.TPSA(assembled),
            "ArRings": Descriptors.NumAromaticRings(assembled),
            "SatRings": Descriptors.NumSaturatedRings(assembled),
            "HeavyAtoms": assembled.GetNumHeavyAtoms(),
        }

        # Compute planarity
        try:
            oop = compute_planarity_score(assembled, parent_3d, match_3d)
            props["max_oop"] = oop["max_oop"]
            props["mean_oop"] = oop["mean_oop"]
        except Exception:
            props["max_oop"] = None
            props["mean_oop"] = None

        # Set properties on mol
        assembled.SetProp("_Name", f"2nd_order_{rank:03d}")
        assembled.SetProp("geo_score", f"{h['score']:.4f}")
        assembled.SetProp("linker", h["linker"])
        assembled.SetProp("orig_fragment", h["orig_smi"])
        assembled.SetProp("ext_fragment", h["ext_smi"])
        for k, v in props.items():
            if v is not None:
                assembled.SetProp(k, f"{v:.2f}" if isinstance(v, float) else str(v))

        # Write individual files
        w = Chem.SDWriter(f"{OUTDIR}/assembled/assembled_{rank:03d}.sdf")
        w.write(assembled)
        w.close()

        # Write fragment
        mol3d.SetProp("_Name", f"frag_{rank:03d}")
        w2 = Chem.SDWriter(f"{OUTDIR}/fragments/frag_{rank:03d}.sdf")
        w2.write(mol3d, confId=h["conf_idx"])
        w2.close()

        all_assembled_writer.write(assembled)
        all_frag_writer.write(mol3d, confId=h["conf_idx"])

        oop_str = f"{props['max_oop']:.2f}" if props['max_oop'] is not None else "N/A"
        status = "OK"

        assembled_results.append({
            "rank": rank, "ext_smi": h["ext_smi"], "orig_smi": h["orig_smi"],
            "linker": h["linker"], "score": h["score"],
            "frag_hba": h.get("frag_hba", ""),
            "frag_hbd": h.get("frag_hbd", ""),
            "status": status,
            "assembled_smi": assembled_smi,
            "max_oop": props.get("max_oop"),
            **props,
        })

        if rank <= 50 or rank % 50 == 0:
            print(f"  {rank:>3} {status:<4} {h['score']:>6.3f}  OOP:{oop_str:>5}  "
                  f"fHBA:{h.get('frag_hba','?'):>1}  {props['MW']:>6.1f}  "
                  f"{h['linker']:>8}  {h['ext_smi'][:45]}")

    all_assembled_writer.close()
    all_frag_writer.close()

    # ─── 9. Write CSV ────────────────────────────────────────────
    csv_path = f"{OUTDIR}/properties.csv"
    if assembled_results:
        keys = ["rank", "orig_smi", "linker", "ext_smi", "assembled_smi",
                "score", "frag_hba", "frag_hbd", "status", "max_oop", "mean_oop",
                "MW", "cLogP", "HBA", "HBD", "RotBonds", "TPSA",
                "ArRings", "SatRings", "HeavyAtoms"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in assembled_results:
                writer.writerow(r)
        print(f"\n  CSV written to {csv_path}")

    # ─── 10. Summary ─────────────────────────────────────────────
    n_ok = sum(1 for r in assembled_results if r["status"] == "OK")
    n_fail = sum(1 for r in assembled_results if r["status"] != "OK")
    n_good_oop = sum(1 for r in assembled_results
                     if r["status"] == "OK" and r.get("max_oop") is not None
                     and r["max_oop"] < 0.3)

    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Total time: {elapsed_total/60:.1f} min")
    print(f"  Assembled: {n_ok} OK, {n_fail} failed")
    print(f"  Good planarity (OOP < 0.3): {n_good_oop}")
    print(f"  Direct screening comparison: {len(direct_results)} direct vs "
          f"{n_ok} novel 2nd-order")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
