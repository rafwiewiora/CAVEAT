"""SQLite fragment database for CAVEAT.

Stores fragments with their 3D conformers and geometric descriptors.
Supports building from molecules, querying by geometry, and KDTree-based
nearest-neighbor search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

from caveat.fragment import Fragment, AttachmentPoint, BRICSFragmenter, Fragmenter
from caveat.geometry import (
    embed_fragment,
    compute_vector_pair_descriptor,
    VectorPairDescriptor,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_smiles TEXT UNIQUE NOT NULL,
    mol_binary BLOB,
    num_heavy_atoms INTEGER,
    num_attachment_points INTEGER,
    brics_labels TEXT,
    attachment_points_json TEXT,
    source_count INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS conformers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id INTEGER NOT NULL,
    conformer_idx INTEGER NOT NULL,
    energy REAL,
    FOREIGN KEY (fragment_id) REFERENCES fragments(id)
);

CREATE TABLE IF NOT EXISTS attachment_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id INTEGER NOT NULL,
    conformer_id INTEGER NOT NULL,
    ap_index INTEGER NOT NULL,
    origin_x REAL, origin_y REAL, origin_z REAL,
    direction_x REAL, direction_y REAL, direction_z REAL,
    FOREIGN KEY (fragment_id) REFERENCES fragments(id),
    FOREIGN KEY (conformer_id) REFERENCES conformers(id)
);

CREATE TABLE IF NOT EXISTS vector_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id INTEGER NOT NULL,
    conformer_id INTEGER NOT NULL,
    ap1_index INTEGER NOT NULL,
    ap2_index INTEGER NOT NULL,
    distance REAL NOT NULL,
    angle1 REAL NOT NULL,
    angle2 REAL NOT NULL,
    dihedral REAL NOT NULL,
    FOREIGN KEY (fragment_id) REFERENCES fragments(id),
    FOREIGN KEY (conformer_id) REFERENCES conformers(id)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id INTEGER NOT NULL,
    source_smiles TEXT NOT NULL,
    FOREIGN KEY (fragment_id) REFERENCES fragments(id)
);

CREATE INDEX IF NOT EXISTS idx_vector_pairs_geom
ON vector_pairs(distance, angle1, angle2, dihedral);

CREATE INDEX IF NOT EXISTS idx_fragments_nap
ON fragments(num_attachment_points);

CREATE INDEX IF NOT EXISTS idx_vector_pairs_frag
ON vector_pairs(fragment_id);
"""


def _fragment_worker(args):
    """Worker for parallel BRICS fragmentation.

    Accepts a batch of SMILES strings, parses, fragments, and deduplicates
    locally within the chunk. Returns a dict of unique fragments.
    """
    smiles_batch, min_heavy_atoms = args
    fragmenter = BRICSFragmenter(min_heavy_atoms=min_heavy_atoms)

    # Local dedup within this chunk
    unique = {}  # smiles → (mol_binary, ap_dicts, labels, nha, nap, [sources])

    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        frags = fragmenter.fragment(mol, source_smiles=smi)
        for frag in frags:
            if frag.smiles in unique:
                unique[frag.smiles][-1].append(smi)
            else:
                unique[frag.smiles] = [
                    frag.mol.ToBinary(),
                    [ap.to_dict() for ap in frag.attachment_points],
                    frag.brics_labels,
                    frag.num_heavy_atoms,
                    frag.num_attachment_points,
                    [smi],
                ]
    return unique


def _embed_worker(args):
    """Worker function for parallel fragment embedding.

    Accepts (smiles, mol_binary, n_confs) and returns a dict with
    the embedded mol binary and extracted geometric data, or None on failure.
    """
    smiles, mol_binary, ap_dicts, n_confs = args
    mol = Chem.Mol(mol_binary)
    try:
        mol3d = embed_fragment(mol, n_confs=n_confs)
    except Exception:
        return None

    aps = _find_attachment_points_in_mol(mol3d)
    conformers = []
    for conf_idx in range(mol3d.GetNumConformers()):
        conf = mol3d.GetConformer(conf_idx)
        energy = _get_conformer_energy(mol3d, conf_idx)

        vectors = []
        for ap_idx, ap in enumerate(aps):
            pos_neighbor = np.array(conf.GetAtomPosition(ap.neighbor_atom_idx))
            pos_dummy = np.array(conf.GetAtomPosition(ap.dummy_atom_idx))
            direction = pos_dummy - pos_neighbor
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            vectors.append((ap_idx,
                            pos_neighbor.tolist(), direction.tolist()))

        pairs = []
        if len(aps) >= 2:
            for (ai, ap1), (aj, ap2) in combinations(enumerate(aps), 2):
                desc = compute_vector_pair_descriptor(
                    conf,
                    ap1.dummy_atom_idx, ap1.neighbor_atom_idx,
                    ap2.dummy_atom_idx, ap2.neighbor_atom_idx,
                )
                pairs.append((ai, aj,
                              desc.distance, desc.angle1,
                              desc.angle2, desc.dihedral))

        conformers.append((conf_idx, energy, vectors, pairs))

    return {
        "smiles": smiles,
        "mol3d_binary": mol3d.ToBinary(),
        "conformers": conformers,
    }


class FragmentDatabase:
    """SQLite-backed fragment database with geometric indexing."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._kdtree = None
        self._kdtree_data = None

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def build(
        self,
        molecules: list[tuple[str, Chem.Mol]],
        fragmenter: Optional[Fragmenter] = None,
        n_confs: int = 10,
        progress_callback=None,
        workers: int = 1,
    ) -> dict:
        """Build the database from a list of (smiles, mol) pairs.

        Args:
            molecules: list of (smiles, mol) tuples
            fragmenter: fragmentation method (default: BRICSFragmenter)
            n_confs: number of conformers per fragment
            progress_callback: optional callable(current, total)
            workers: number of parallel workers for embedding (default: 1)

        Returns:
            dict with build statistics
        """
        if fragmenter is None:
            fragmenter = BRICSFragmenter()

        if workers > 1:
            return self._build_parallel(
                molecules, fragmenter, n_confs, progress_callback, workers,
            )

        stats = {"molecules": 0, "fragments": 0, "conformers": 0, "skipped": 0}
        total = len(molecules)

        for i, (smi, mol) in enumerate(molecules):
            if mol is None:
                stats["skipped"] += 1
                continue

            stats["molecules"] += 1
            frags = fragmenter.fragment(mol, source_smiles=smi)

            for frag in frags:
                frag_id = self._insert_fragment(frag)
                if frag_id is None:
                    # Already exists — increment source count and add source
                    self._add_source(frag.smiles, smi)
                    continue

                stats["fragments"] += 1

                # Add source
                self.conn.execute(
                    "INSERT INTO sources (fragment_id, source_smiles) VALUES (?, ?)",
                    (frag_id, smi),
                )

                # Embed in 3D and compute geometry
                try:
                    mol3d = embed_fragment(frag.mol, n_confs=n_confs)
                except Exception as e:
                    logger.warning(f"Could not embed {frag.smiles}: {e}")
                    continue

                # Recompute attachment points for the Hs-added mol
                aps = _find_attachment_points_in_mol(mol3d)

                for conf_idx in range(mol3d.GetNumConformers()):
                    conf = mol3d.GetConformer(conf_idx)
                    energy = _get_conformer_energy(mol3d, conf_idx)

                    cursor = self.conn.execute(
                        "INSERT INTO conformers (fragment_id, conformer_idx, energy) VALUES (?, ?, ?)",
                        (frag_id, conf_idx, energy),
                    )
                    conf_id = cursor.lastrowid
                    stats["conformers"] += 1

                    # Store exit vectors
                    for ap_idx, ap in enumerate(aps):
                        pos_neighbor = np.array(conf.GetAtomPosition(ap.neighbor_atom_idx))
                        pos_dummy = np.array(conf.GetAtomPosition(ap.dummy_atom_idx))
                        direction = pos_dummy - pos_neighbor
                        norm = np.linalg.norm(direction)
                        if norm > 1e-6:
                            direction = direction / norm

                        self.conn.execute(
                            """INSERT INTO attachment_vectors
                            (fragment_id, conformer_id, ap_index,
                             origin_x, origin_y, origin_z,
                             direction_x, direction_y, direction_z)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (frag_id, conf_id, ap_idx,
                             float(pos_neighbor[0]), float(pos_neighbor[1]), float(pos_neighbor[2]),
                             float(direction[0]), float(direction[1]), float(direction[2])),
                        )

                    # Compute vector pair descriptors for fragments with 2+ APs
                    if len(aps) >= 2:
                        for (ai, ap1), (aj, ap2) in combinations(enumerate(aps), 2):
                            desc = compute_vector_pair_descriptor(
                                conf,
                                ap1.dummy_atom_idx, ap1.neighbor_atom_idx,
                                ap2.dummy_atom_idx, ap2.neighbor_atom_idx,
                            )
                            self.conn.execute(
                                """INSERT INTO vector_pairs
                                (fragment_id, conformer_id, ap1_index, ap2_index,
                                 distance, angle1, angle2, dihedral)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (frag_id, conf_id, ai, aj,
                                 desc.distance, desc.angle1, desc.angle2, desc.dihedral),
                            )

                    # Update mol_binary with the 3D-embedded version (after first conformer only)
                    if conf_idx == 0:
                        mol_binary = mol3d.ToBinary()
                        self.conn.execute(
                            "UPDATE fragments SET mol_binary = ? WHERE id = ?",
                            (mol_binary, frag_id),
                        )

            if progress_callback:
                progress_callback(i + 1, total)

            # Commit periodically
            if (i + 1) % 100 == 0:
                self.conn.commit()

        self.conn.commit()
        self._kdtree = None  # invalidate cache
        return stats

    def _build_parallel(
        self,
        molecules: list[tuple[str, Chem.Mol]],
        fragmenter: Fragmenter,
        n_confs: int,
        progress_callback,
        workers: int,
    ) -> dict:
        """Fully parallel build: fragment in parallel → embed in parallel → insert.

        Phase 1: Fragment molecules in parallel (multiprocessing)
        Phase 2: Embed unique fragments in parallel (multiprocessing)
        Phase 3: Insert results into database (single-threaded, fast)
        """
        stats = {"molecules": 0, "fragments": 0, "conformers": 0, "skipped": 0}

        # Extract SMILES for workers (Mol objects aren't picklable;
        # workers parse SMILES themselves, so None mols are fine)
        smiles_list = [smi for smi, _ in molecules]
        stats["molecules"] = len(smiles_list)

        min_heavy = getattr(fragmenter, "min_heavy_atoms", 3)

        # --- Phase 1: Parallel fragmentation ---
        chunk_size = max(1000, len(smiles_list) // workers)
        chunks = [smiles_list[i:i + chunk_size]
                  for i in range(0, len(smiles_list), chunk_size)]

        if progress_callback:
            progress_callback(0, len(smiles_list), "fragmenting")

        # unique_frags: smiles → [mol_binary, ap_dicts, labels, nha, nap, [sources]]
        unique_frags = {}
        chunks_done = 0
        with Pool(processes=workers) as pool:
            for chunk_result in pool.imap_unordered(
                _fragment_worker,
                [(chunk, min_heavy) for chunk in chunks],
            ):
                for smiles, data in chunk_result.items():
                    if smiles in unique_frags:
                        unique_frags[smiles][-1].extend(data[-1])
                    else:
                        unique_frags[smiles] = data
                chunks_done += 1
                if progress_callback:
                    done_mols = sum(len(chunks[i]) for i in range(chunks_done))
                    progress_callback(
                        min(done_mols, len(smiles_list)),
                        len(smiles_list), "fragmenting",
                    )

        if progress_callback:
            progress_callback(len(smiles_list), len(smiles_list), "fragmenting")

        logger.info(
            f"Phase 1 complete: {len(unique_frags)} unique fragments "
            f"from {stats['molecules']} molecules"
        )

        # --- Phase 2: Parallel embedding ---
        embed_tasks = []
        for smiles, data in unique_frags.items():
            mol_binary, ap_dicts = data[0], data[1]
            embed_tasks.append((smiles, mol_binary, ap_dicts, n_confs))

        embed_results = {}
        n_embedded = 0
        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(
                _embed_worker, embed_tasks, chunksize=32,
            ):
                n_embedded += 1
                if progress_callback and n_embedded % 500 == 0:
                    progress_callback(n_embedded, len(embed_tasks), "embedding")
                if result is not None:
                    embed_results[result["smiles"]] = result

        if progress_callback:
            progress_callback(len(embed_tasks), len(embed_tasks), "embedding")

        logger.info(
            f"Phase 2 complete: embedded {len(embed_results)}/{len(unique_frags)} fragments"
        )

        # --- Phase 3: Insert into database (serial, fast) ---
        n_inserted = 0
        for smiles, data in unique_frags.items():
            mol_binary, ap_dicts, labels, nha, nap, sources = data
            labels_json = json.dumps(labels)
            ap_json = json.dumps(ap_dicts)

            try:
                cursor = self.conn.execute(
                    """INSERT INTO fragments
                    (canonical_smiles, num_heavy_atoms, num_attachment_points,
                     brics_labels, attachment_points_json, source_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (smiles, nha, nap, labels_json, ap_json, len(sources)),
                )
                frag_id = cursor.lastrowid
            except sqlite3.IntegrityError:
                continue

            stats["fragments"] += 1

            for src in sources:
                self.conn.execute(
                    "INSERT INTO sources (fragment_id, source_smiles) VALUES (?, ?)",
                    (frag_id, src),
                )

            embed_data = embed_results.get(smiles)
            if embed_data is None:
                continue

            self.conn.execute(
                "UPDATE fragments SET mol_binary = ? WHERE id = ?",
                (embed_data["mol3d_binary"], frag_id),
            )

            for conf_idx, energy, vectors, pairs in embed_data["conformers"]:
                cursor = self.conn.execute(
                    "INSERT INTO conformers (fragment_id, conformer_idx, energy) "
                    "VALUES (?, ?, ?)",
                    (frag_id, conf_idx, energy),
                )
                conf_id = cursor.lastrowid
                stats["conformers"] += 1

                for ap_idx, origin, direction in vectors:
                    self.conn.execute(
                        """INSERT INTO attachment_vectors
                        (fragment_id, conformer_id, ap_index,
                         origin_x, origin_y, origin_z,
                         direction_x, direction_y, direction_z)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (frag_id, conf_id, ap_idx,
                         origin[0], origin[1], origin[2],
                         direction[0], direction[1], direction[2]),
                    )

                for ap1_idx, ap2_idx, dist, a1, a2, dih in pairs:
                    self.conn.execute(
                        """INSERT INTO vector_pairs
                        (fragment_id, conformer_id, ap1_index, ap2_index,
                         distance, angle1, angle2, dihedral)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (frag_id, conf_id, ap1_idx, ap2_idx,
                         dist, a1, a2, dih),
                    )

            n_inserted += 1
            if n_inserted % 1000 == 0:
                self.conn.commit()
                if progress_callback:
                    progress_callback(n_inserted, len(unique_frags), "inserting")

        self.conn.commit()
        self._kdtree = None

        if progress_callback:
            progress_callback(len(unique_frags), len(unique_frags), "inserting")

        return stats

    def _insert_fragment(self, frag: Fragment) -> Optional[int]:
        """Insert a fragment, return its ID, or None if already exists."""
        ap_json = json.dumps([ap.to_dict() for ap in frag.attachment_points])
        labels_json = json.dumps(frag.brics_labels)
        try:
            cursor = self.conn.execute(
                """INSERT INTO fragments
                (canonical_smiles, num_heavy_atoms, num_attachment_points,
                 brics_labels, attachment_points_json, source_count)
                VALUES (?, ?, ?, ?, ?, 1)""",
                (frag.smiles, frag.num_heavy_atoms, frag.num_attachment_points,
                 labels_json, ap_json),
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def _add_source(self, frag_smiles: str, source_smiles: str):
        """Add a source molecule for an existing fragment."""
        row = self.conn.execute(
            "SELECT id FROM fragments WHERE canonical_smiles = ?",
            (frag_smiles,),
        ).fetchone()
        if row:
            frag_id = row[0]
            self.conn.execute(
                "UPDATE fragments SET source_count = source_count + 1 WHERE id = ?",
                (frag_id,),
            )
            self.conn.execute(
                "INSERT INTO sources (fragment_id, source_smiles) VALUES (?, ?)",
                (frag_id, source_smiles),
            )

    def query_by_geometry(
        self,
        distance: float,
        angle1: float,
        angle2: float,
        dihedral: float,
        tolerances: Optional[dict] = None,
        num_attachment_points: Optional[int] = None,
    ) -> list[dict]:
        """Query for fragments matching geometric descriptors within tolerances.

        Args:
            distance, angle1, angle2, dihedral: target geometry
            tolerances: dict with keys 'distance', 'angle1', 'angle2', 'dihedral'
            num_attachment_points: filter by number of attachment points

        Returns:
            list of dicts with fragment_id, conformer_id, geometric distance
        """
        if tolerances is None:
            tolerances = {
                "distance": 0.5,
                "angle1": 15.0,
                "angle2": 15.0,
                "dihedral": 30.0,
            }

        query = """
            SELECT vp.fragment_id, vp.conformer_id,
                   vp.distance, vp.angle1, vp.angle2, vp.dihedral,
                   f.canonical_smiles, f.num_attachment_points
            FROM vector_pairs vp
            JOIN fragments f ON vp.fragment_id = f.id
            WHERE vp.distance BETWEEN ? AND ?
              AND vp.angle1 BETWEEN ? AND ?
              AND vp.angle2 BETWEEN ? AND ?
              AND vp.dihedral BETWEEN ? AND ?
        """
        params = [
            distance - tolerances["distance"], distance + tolerances["distance"],
            angle1 - tolerances["angle1"], angle1 + tolerances["angle1"],
            angle2 - tolerances["angle2"], angle2 + tolerances["angle2"],
            dihedral - tolerances["dihedral"], dihedral + tolerances["dihedral"],
        ]

        if num_attachment_points is not None:
            query += " AND f.num_attachment_points = ?"
            params.append(num_attachment_points)

        rows = self.conn.execute(query, params).fetchall()

        results = []
        target = np.array([distance, angle1, angle2, dihedral])
        for row in rows:
            frag_id, conf_id, d, a1, a2, dih, smi, nap = row
            geom = np.array([d, a1, a2, dih])
            # Weighted distance (normalize angles by scale)
            diff = target - geom
            # Weight: distance in Å, angles in degrees
            weights = np.array([1.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 30.0])
            geo_dist = float(np.linalg.norm(diff * weights))
            results.append({
                "fragment_id": frag_id,
                "conformer_id": conf_id,
                "smiles": smi,
                "distance": d,
                "angle1": a1,
                "angle2": a2,
                "dihedral": dih,
                "geometric_distance": geo_dist,
                "num_attachment_points": nap,
            })

        results.sort(key=lambda x: x["geometric_distance"])
        return results

    def query_by_kdtree(
        self,
        distance: float,
        angle1: float,
        angle2: float,
        dihedral: float,
        k: int = 20,
        num_attachment_points: Optional[int] = None,
    ) -> list[dict]:
        """Query using KDTree for fast nearest-neighbor search in 4D geometry space."""
        from scipy.spatial import KDTree

        if self._kdtree is None:
            self._build_kdtree()

        if self._kdtree is None or len(self._kdtree_data) == 0:
            return []

        # Normalize query point to match KDTree scaling
        query_point = np.array([
            distance,
            angle1 / 15.0,
            angle2 / 15.0,
            dihedral / 30.0,
        ])

        distances, indices = self._kdtree.query(query_point, k=min(k * 3, len(self._kdtree_data)))

        if isinstance(distances, float):
            distances = [distances]
            indices = [indices]

        results = []
        seen_frags = set()
        for dist, idx in zip(distances, indices):
            if idx >= len(self._kdtree_data):
                continue
            entry = self._kdtree_data[idx]
            frag_id = entry["fragment_id"]

            if num_attachment_points is not None:
                if entry["num_attachment_points"] != num_attachment_points:
                    continue

            if frag_id in seen_frags:
                continue
            seen_frags.add(frag_id)

            results.append({
                "fragment_id": frag_id,
                "conformer_id": entry["conformer_id"],
                "smiles": entry["smiles"],
                "distance": entry["distance"],
                "angle1": entry["angle1"],
                "angle2": entry["angle2"],
                "dihedral": entry["dihedral"],
                "geometric_distance": float(dist),
                "num_attachment_points": entry["num_attachment_points"],
            })

            if len(results) >= k:
                break

        return results

    def _build_kdtree(self):
        from scipy.spatial import KDTree

        rows = self.conn.execute("""
            SELECT vp.fragment_id, vp.conformer_id,
                   vp.distance, vp.angle1, vp.angle2, vp.dihedral,
                   f.canonical_smiles, f.num_attachment_points
            FROM vector_pairs vp
            JOIN fragments f ON vp.fragment_id = f.id
        """).fetchall()

        if not rows:
            self._kdtree = None
            self._kdtree_data = []
            return

        points = []
        self._kdtree_data = []
        for row in rows:
            frag_id, conf_id, d, a1, a2, dih, smi, nap = row
            # Normalize to make distances comparable
            points.append([d, a1 / 15.0, a2 / 15.0, dih / 30.0])
            self._kdtree_data.append({
                "fragment_id": frag_id,
                "conformer_id": conf_id,
                "smiles": smi,
                "distance": d,
                "angle1": a1,
                "angle2": a2,
                "dihedral": dih,
                "num_attachment_points": nap,
            })

        self._kdtree = KDTree(np.array(points))

    def get_fragment(self, fragment_id: int) -> Optional[dict]:
        """Retrieve a fragment by ID, including its mol object."""
        row = self.conn.execute(
            """SELECT id, canonical_smiles, mol_binary, num_heavy_atoms,
                      num_attachment_points, brics_labels, attachment_points_json,
                      source_count
               FROM fragments WHERE id = ?""",
            (fragment_id,),
        ).fetchone()

        if row is None:
            return None

        fid, smi, mol_bin, nha, nap, labels_json, ap_json, sc = row
        mol = None
        if mol_bin:
            mol = Chem.Mol(mol_bin)

        aps = [AttachmentPoint.from_dict(d) for d in json.loads(ap_json)]

        return {
            "id": fid,
            "smiles": smi,
            "mol": mol,
            "num_heavy_atoms": nha,
            "num_attachment_points": nap,
            "brics_labels": json.loads(labels_json),
            "attachment_points": aps,
            "source_count": sc,
        }

    def get_attachment_vectors(self, fragment_id: int, conformer_id: int) -> list[dict]:
        """Retrieve stored exit vector data for a fragment conformer.

        Args:
            fragment_id: the fragment's database ID
            conformer_id: the conformer's database ID

        Returns:
            list of dicts with 'ap_index', 'origin' (np.ndarray), 'direction' (np.ndarray)
        """
        rows = self.conn.execute(
            """SELECT ap_index, origin_x, origin_y, origin_z,
                      direction_x, direction_y, direction_z
               FROM attachment_vectors
               WHERE fragment_id = ? AND conformer_id = ?
               ORDER BY ap_index""",
            (fragment_id, conformer_id),
        ).fetchall()

        result = []
        for row in rows:
            ap_idx, ox, oy, oz, dx, dy, dz = row
            result.append({
                "ap_index": ap_idx,
                "origin": np.array([ox, oy, oz]),
                "direction": np.array([dx, dy, dz]),
            })
        return result

    def get_sources(self, fragment_id: int) -> list[str]:
        """Get source molecule SMILES for a fragment."""
        rows = self.conn.execute(
            "SELECT source_smiles FROM sources WHERE fragment_id = ?",
            (fragment_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}
        stats["num_fragments"] = self.conn.execute(
            "SELECT COUNT(*) FROM fragments"
        ).fetchone()[0]
        stats["num_conformers"] = self.conn.execute(
            "SELECT COUNT(*) FROM conformers"
        ).fetchone()[0]
        stats["num_vector_pairs"] = self.conn.execute(
            "SELECT COUNT(*) FROM vector_pairs"
        ).fetchone()[0]
        stats["num_sources"] = self.conn.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0]
        stats["num_unique_sources"] = self.conn.execute(
            "SELECT COUNT(DISTINCT source_smiles) FROM sources"
        ).fetchone()[0]

        # Attachment point distribution
        rows = self.conn.execute(
            "SELECT num_attachment_points, COUNT(*) FROM fragments GROUP BY num_attachment_points ORDER BY num_attachment_points"
        ).fetchall()
        stats["ap_distribution"] = {r[0]: r[1] for r in rows}

        # Heavy atom stats
        row = self.conn.execute(
            "SELECT MIN(num_heavy_atoms), MAX(num_heavy_atoms), AVG(num_heavy_atoms) FROM fragments"
        ).fetchone()
        if row[0] is not None:
            stats["min_heavy_atoms"] = row[0]
            stats["max_heavy_atoms"] = row[1]
            stats["avg_heavy_atoms"] = round(row[2], 1)

        return stats


def _find_attachment_points_in_mol(mol: Chem.Mol) -> list[AttachmentPoint]:
    """Find attachment points (dummy atoms) in a mol that may have Hs added."""
    aps = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            isotope = atom.GetIsotope()
            neighbors = atom.GetNeighbors()
            if neighbors:
                aps.append(AttachmentPoint(
                    dummy_atom_idx=atom.GetIdx(),
                    neighbor_atom_idx=neighbors[0].GetIdx(),
                    brics_label=isotope,
                ))
    return aps


def _get_conformer_energy(mol: Chem.Mol, conf_idx: int) -> Optional[float]:
    """Try to get MMFF energy for a conformer."""
    try:
        from rdkit.Chem import rdForceFieldHelpers
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, rdForceFieldHelpers.MMFFGetMoleculeProperties(mol),
            confId=conf_idx,
        )
        if ff:
            return ff.CalcEnergy()
    except Exception:
        pass
    return None
