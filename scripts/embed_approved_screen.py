"""Embed filtered approved drug fragments for screening.

Reads from the fragment-only DB (which may lack mol_binary),
reconstructs mols from SMILES, embeds, and recomputes APs dynamically.
"""
import json
import sqlite3
import numpy as np
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import Descriptors
from caveat.geometry import embed_fragment, compute_vector_pair_descriptor
from caveat.database import _find_attachment_points_in_mol, SCHEMA

src = sqlite3.connect('/Users/rafal/repos/CAVEAT/approved_drugs_frags.db')
rows = src.execute('''
    SELECT canonical_smiles, num_heavy_atoms, num_attachment_points,
           brics_labels, source_count
    FROM fragments
    WHERE num_heavy_atoms <= 14 AND num_attachment_points IN (1, 2)
''').fetchall()

candidates = []
for smi, nha, nap, labels, sc in rows:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    if Descriptors.NumAromaticRings(mol) <= 1:
        candidates.append((smi, mol, nha, nap, labels, sc))
src.close()
print(f'Embedding {len(candidates)} fragments...')

import os
db_path = '/Users/rafal/repos/CAVEAT/approved_drugs_screen.db'
if os.path.exists(db_path):
    os.remove(db_path)
dst = sqlite3.connect(db_path)
dst.execute('PRAGMA journal_mode=WAL')
dst.execute('PRAGMA synchronous=NORMAL')
dst.executescript(SCHEMA)
dst.commit()

embedded = 0
failed = 0
for i, (smi, mol, nha, nap, labels, sc) in enumerate(candidates):
    try:
        mol3d = embed_fragment(mol, n_confs=1)
    except Exception:
        failed += 1
        continue

    # Recompute APs dynamically from the embedded mol (correct indices)
    aps = _find_attachment_points_in_mol(mol3d)
    ap_dicts = [{"dummy_atom_idx": ap.dummy_atom_idx,
                 "neighbor_atom_idx": ap.neighbor_atom_idx,
                 "brics_label": ap.brics_label, "bond_order": 1} for ap in aps]

    cursor = dst.execute(
        """INSERT INTO fragments
        (canonical_smiles, mol_binary, num_heavy_atoms, num_attachment_points,
         brics_labels, attachment_points_json, source_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (smi, mol3d.ToBinary(), nha, nap, labels,
         json.dumps(ap_dicts), sc),
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
    if (i + 1) % 500 == 0:
        dst.commit()
        print(f'  {i+1}/{len(candidates)} embedded...')

dst.commit()
dst.close()
print(f'Done: {embedded} embedded, {failed} failed')
