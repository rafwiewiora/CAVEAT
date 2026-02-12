#!/usr/bin/env python
"""Build a screening DB from a filtered subset of chembl_full_frags.db.

Extracts fragments matching: HA<=14, 2 APs, 0 aromatic rings, <=2 rotatable bonds.
Embeds them with 3D conformers and computes geometric descriptors.
Uses multiprocessing for embedding.
"""
import json
import os
import sqlite3
import time
import multiprocessing as mp
import numpy as np
from itertools import combinations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from caveat.geometry import embed_fragment, compute_vector_pair_descriptor
from caveat.database import _find_attachment_points_in_mol, SCHEMA

RDLogger.DisableLog('rdApp.*')

SRC_DB = '/Users/rafal/repos/CAVEAT/chembl_full_frags.db'
DST_DB = '/Users/rafal/repos/CAVEAT/chembl_screen_rot2.db'
NUM_WORKERS = 14
BATCH_SIZE = 500


def _embed_worker(batch):
    """Embed a batch of (smiles, nha, nap, labels, sc) and return serialized results."""
    results = []
    for smi, nha, nap, labels, sc in batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            mol3d = embed_fragment(mol, n_confs=1)
        except Exception:
            continue
        results.append((smi, mol3d.ToBinary(), nha, nap, labels, sc))
    return results


def main():
    # Read source fragments
    print('Reading source fragments...')
    t0 = time.time()
    src = sqlite3.connect(SRC_DB)
    rows = src.execute('''
        SELECT canonical_smiles, num_heavy_atoms, num_attachment_points,
               brics_labels, source_count
        FROM fragments
        WHERE num_heavy_atoms <= 14
          AND num_attachment_points = 2
          AND num_aromatic_rings = 0
          AND num_rotatable_bonds <= 2
    ''').fetchall()
    src.close()
    print(f'Read {len(rows):,} fragments in {time.time()-t0:.0f}s')

    # Create destination DB
    if os.path.exists(DST_DB):
        os.remove(DST_DB)
    dst = sqlite3.connect(DST_DB)
    dst.execute('PRAGMA journal_mode=WAL')
    dst.execute('PRAGMA synchronous=NORMAL')
    dst.executescript(SCHEMA)
    dst.commit()

    # Split into batches for multiprocessing
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    print(f'Embedding with {NUM_WORKERS} workers, {len(batches)} batches...')

    pool = mp.Pool(NUM_WORKERS)
    embedded = 0
    t1 = time.time()

    for batch_idx, result_batch in enumerate(pool.imap_unordered(_embed_worker, batches)):
        for smi, mol_binary, nha, nap, labels, sc in result_batch:
            mol3d = Chem.Mol(mol_binary)
            aps = _find_attachment_points_in_mol(mol3d)
            ap_dicts = [{"dummy_atom_idx": ap.dummy_atom_idx,
                         "neighbor_atom_idx": ap.neighbor_atom_idx,
                         "brics_label": ap.brics_label, "bond_order": 1} for ap in aps]

            cursor = dst.execute(
                """INSERT INTO fragments
                (canonical_smiles, mol_binary, num_heavy_atoms, num_attachment_points,
                 brics_labels, attachment_points_json, source_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (smi, mol_binary, nha, nap, labels, json.dumps(ap_dicts), sc),
            )
            frag_id = cursor.lastrowid

            for conf_idx in range(mol3d.GetNumConformers()):
                conf = mol3d.GetConformer(conf_idx)
                cur2 = dst.execute(
                    "INSERT INTO conformers (fragment_id, conformer_idx, energy) VALUES (?, ?, ?)",
                    (frag_id, conf_idx, None),
                )
                conf_id = cur2.lastrowid

                for ap_idx, ap in enumerate(aps):
                    pos_n = np.array(conf.GetAtomPosition(ap.neighbor_atom_idx))
                    pos_d = np.array(conf.GetAtomPosition(ap.dummy_atom_idx))
                    direction = pos_d - pos_n
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                    dst.execute(
                        """INSERT INTO attachment_vectors
                        (fragment_id, conformer_id, ap_index, origin_x, origin_y, origin_z,
                         direction_x, direction_y, direction_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (frag_id, conf_id, ap_idx,
                         float(pos_n[0]), float(pos_n[1]), float(pos_n[2]),
                         float(direction[0]), float(direction[1]), float(direction[2])),
                    )

                if len(aps) >= 2:
                    for (ai, ap1), (aj, ap2) in combinations(enumerate(aps), 2):
                        desc = compute_vector_pair_descriptor(
                            conf, ap1.dummy_atom_idx, ap1.neighbor_atom_idx,
                            ap2.dummy_atom_idx, ap2.neighbor_atom_idx)
                        dst.execute(
                            """INSERT INTO vector_pairs
                            (fragment_id, conformer_id, ap1_index, ap2_index,
                             distance, angle1, angle2, dihedral) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (frag_id, conf_id, ai, aj,
                             desc.distance, desc.angle1, desc.angle2, desc.dihedral),
                        )

            embedded += 1

        if (batch_idx + 1) % 20 == 0:
            dst.commit()
            elapsed = time.time() - t1
            rate = embedded / elapsed if elapsed > 0 else 0
            total_est = len(rows)
            eta = (total_est - embedded) / rate if rate > 0 else 0
            print(f'  {embedded:>8,} / ~{total_est:,}  ({embedded*100/total_est:.1f}%)  '
                  f'{rate:.0f}/s  ETA: {eta/60:.1f} min')

    pool.close()
    pool.join()
    dst.commit()

    # Build KDTree index
    print('Building KDTree index...')
    from caveat.database import FragmentDatabase
    dst.close()
    db = FragmentDatabase(DST_DB)
    db._build_kdtree()
    db.close()

    elapsed = time.time() - t0
    print(f'\nDone! {embedded:,} fragments embedded in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
