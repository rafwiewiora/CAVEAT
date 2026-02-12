#!/usr/bin/env python
"""Build a fragment-only DB using hierarchical BRICS fragmentation.

Phase 1: Fragment all ChEMBL drug-like molecules with HierarchicalBRICSFragmenter
         (max_cuts=2), streaming to SQLite. No 3D embedding.
Phase 2: Use embed_chembl_screen.py (or similar) to extract and embed a filtered subset.

This script is restartable: if the DB exists, it checks which molecule batches have
been processed (via a progress table) and resumes from where it left off.

Output: A fragment-only DB (~1-3GB) with all unique hierarchical fragments.
Estimated: ~20-25M unique fragments from 2.2M molecules, ~30-60 min with 14 workers.
"""
import os
import sqlite3
import sys
import time
import multiprocessing as mp
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from caveat.fragment import HierarchicalBRICSFragmenter
from caveat.database import SCHEMA

RDLogger.DisableLog('rdApp.*')

INPUT_FILE = '/Users/rafal/repos/CAVEAT/data/chembl_druglike.smi'
DST_DB = '/Users/rafal/repos/CAVEAT/chembl_hier_frags.db'
NUM_WORKERS = 14
CHUNK_SIZE = 5000  # molecules per worker chunk
MAX_CUTS = 2
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 14  # pre-filter during fragmentation to reduce DB size
COMMIT_EVERY = 5  # commit every N chunks


# Extra columns for filtering later (computed once during fragmentation)
EXTRA_COLUMNS = """
ALTER TABLE fragments ADD COLUMN num_rotatable_bonds INTEGER;
ALTER TABLE fragments ADD COLUMN num_aromatic_rings INTEGER;
"""


def _fragment_chunk(args):
    """Worker: fragment a chunk of SMILES, return (chunk_id, dict of unique fragments).

    Returns: (chunk_id, {smiles: (nha, nap, labels, source_count, num_rot_bonds, num_ar_rings)})
    """
    chunk_id, smiles_list, min_ha, max_cuts, max_ha = args
    fragmenter = HierarchicalBRICSFragmenter(min_heavy_atoms=min_ha, max_cuts=max_cuts)

    unique = {}  # smiles -> (nha, nap, labels_json, count, nrb, nar)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        frags = fragmenter.fragment(mol, source_smiles=smi)
        for frag in frags:
            if max_ha is not None and frag.num_heavy_atoms > max_ha:
                continue
            if frag.smiles in unique:
                # increment source count
                data = unique[frag.smiles]
                unique[frag.smiles] = (data[0], data[1], data[2], data[3] + 1, data[4], data[5])
            else:
                # Compute descriptors once
                fmol = Chem.MolFromSmiles(frag.smiles)
                if fmol is None:
                    nrb, nar = 0, 0
                else:
                    nrb = Descriptors.NumRotatableBonds(fmol)
                    nar = Descriptors.NumAromaticRings(fmol)

                import json
                labels_json = json.dumps(frag.brics_labels)
                unique[frag.smiles] = (
                    frag.num_heavy_atoms,
                    frag.num_attachment_points,
                    labels_json,
                    1,
                    nrb,
                    nar,
                )

    return chunk_id, unique


def main():
    # Read input SMILES
    print('Reading input SMILES...')
    t0 = time.time()
    smiles_list = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            smiles_list.append(line.split()[0])
    print(f'Read {len(smiles_list):,} molecules in {time.time()-t0:.1f}s')

    # Create/open DB
    is_resume = os.path.exists(DST_DB)
    dst = sqlite3.connect(DST_DB)
    dst.execute('PRAGMA journal_mode=WAL')
    dst.execute('PRAGMA synchronous=NORMAL')
    dst.execute('PRAGMA cache_size=-256000')  # 256MB cache

    if not is_resume:
        # Create fragment schema (we only use the fragments table, no conformers/vectors)
        dst.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_smiles TEXT UNIQUE NOT NULL,
                num_heavy_atoms INTEGER,
                num_attachment_points INTEGER,
                brics_labels TEXT,
                source_count INTEGER DEFAULT 1,
                num_rotatable_bonds INTEGER,
                num_aromatic_rings INTEGER
            )
        """)
        dst.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                chunk_id INTEGER PRIMARY KEY,
                n_molecules INTEGER,
                n_new_fragments INTEGER,
                completed_at REAL
            )
        """)
        dst.commit()
    else:
        # Check if progress table exists
        tables = [r[0] for r in dst.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if 'progress' not in tables:
            dst.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    chunk_id INTEGER PRIMARY KEY,
                    n_molecules INTEGER,
                    n_new_fragments INTEGER,
                    completed_at REAL
                )
            """)
            dst.commit()

    # Determine which chunks are already done
    done_chunks = set(
        r[0] for r in dst.execute('SELECT chunk_id FROM progress').fetchall()
    )
    total_frags = dst.execute('SELECT COUNT(*) FROM fragments').fetchone()[0]

    # Split into chunks
    chunks = []
    for i in range(0, len(smiles_list), CHUNK_SIZE):
        chunk_id = i // CHUNK_SIZE
        if chunk_id not in done_chunks:
            chunks.append((chunk_id, smiles_list[i:i + CHUNK_SIZE]))

    total_chunks = (len(smiles_list) + CHUNK_SIZE - 1) // CHUNK_SIZE

    if is_resume:
        print(f'Resuming: {len(done_chunks)}/{total_chunks} chunks done, '
              f'{total_frags:,} fragments in DB, {len(chunks)} chunks remaining')
    else:
        print(f'Starting fresh: {total_chunks} chunks of {CHUNK_SIZE} molecules')

    if not chunks:
        print('All chunks done!')
        _print_stats(dst)
        dst.close()
        return

    # Process with multiprocessing
    print(f'Fragmenting with {NUM_WORKERS} workers (hierarchical, max_cuts={MAX_CUTS})...')
    t1 = time.time()
    chunks_done_session = 0
    new_frags_session = 0

    # Prepare worker args
    worker_args = [
        (chunk_id, chunk_smiles, MIN_HEAVY_ATOMS, MAX_CUTS, MAX_HEAVY_ATOMS)
        for chunk_id, chunk_smiles in chunks
    ]

    pool = mp.Pool(NUM_WORKERS)
    pending_commits = 0

    for chunk_id, result in pool.imap_unordered(
        _fragment_chunk, worker_args, chunksize=1
    ):
        n_new = 0

        for smiles, (nha, nap, labels_json, count, nrb, nar) in result.items():
            try:
                dst.execute(
                    """INSERT INTO fragments
                    (canonical_smiles, num_heavy_atoms, num_attachment_points,
                     brics_labels, source_count, num_rotatable_bonds, num_aromatic_rings)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (smiles, nha, nap, labels_json, count, nrb, nar),
                )
                n_new += 1
            except sqlite3.IntegrityError:
                # Already exists — increment source count
                dst.execute(
                    "UPDATE fragments SET source_count = source_count + ? "
                    "WHERE canonical_smiles = ?",
                    (count, smiles),
                )

        # Record progress
        dst.execute(
            "INSERT INTO progress (chunk_id, n_molecules, n_new_fragments, completed_at) "
            "VALUES (?, ?, ?, ?)",
            (chunk_id, CHUNK_SIZE, n_new, time.time()),
        )

        chunks_done_session += 1
        new_frags_session += n_new
        total_frags += n_new
        pending_commits += 1

        if pending_commits >= COMMIT_EVERY:
            dst.commit()
            pending_commits = 0

        elapsed = time.time() - t1
        rate = chunks_done_session / elapsed if elapsed > 0 else 0
        remaining = len(chunks) - chunks_done_session
        eta = remaining / rate if rate > 0 else 0

        total_done = len(done_chunks) + chunks_done_session
        print(f'  Chunk {total_done}/{total_chunks}  '
              f'+{n_new:,} new ({total_frags:,} total)  '
              f'{rate:.1f} chunks/s  '
              f'ETA: {eta/60:.1f} min  '
              f'DB: {os.path.getsize(DST_DB)/1e9:.2f} GB')

    pool.close()
    pool.join()
    dst.commit()

    # Checkpoint WAL
    print('Checkpointing WAL...')
    dst.execute('PRAGMA wal_checkpoint(TRUNCATE)')

    # Build index
    print('Building indexes...')
    dst.execute(
        'CREATE INDEX IF NOT EXISTS idx_hier_filter '
        'ON fragments(num_heavy_atoms, num_attachment_points, '
        'num_aromatic_rings, num_rotatable_bonds)'
    )
    dst.commit()

    elapsed_total = time.time() - t0
    print(f'\nDone in {elapsed_total/60:.1f} min')
    print(f'  Session: +{new_frags_session:,} new fragments in {chunks_done_session} chunks')

    _print_stats(dst)
    dst.close()


def _print_stats(dst):
    """Print summary statistics."""
    total = dst.execute('SELECT COUNT(*) FROM fragments').fetchone()[0]
    print(f'\nDatabase statistics:')
    print(f'  Total fragments: {total:,}')

    for nap in [1, 2, 3]:
        count = dst.execute(
            'SELECT COUNT(*) FROM fragments WHERE num_attachment_points = ?', (nap,)
        ).fetchone()[0]
        print(f'  {nap}-AP: {count:,}')

    # Screen-eligible (2-AP, HA<=14, 0-ArRings, RotB<=1)
    screen = dst.execute(
        'SELECT COUNT(*) FROM fragments WHERE num_attachment_points = 2 '
        'AND num_heavy_atoms <= 14 AND num_aromatic_rings = 0 '
        'AND num_rotatable_bonds <= 1'
    ).fetchone()[0]
    print(f'  Screen-eligible (2AP, HA<=14, 0Ar, RotB<=1): {screen:,}')

    screen2 = dst.execute(
        'SELECT COUNT(*) FROM fragments WHERE num_attachment_points = 2 '
        'AND num_heavy_atoms <= 14 AND num_aromatic_rings = 0 '
        'AND num_rotatable_bonds <= 2'
    ).fetchone()[0]
    print(f'  Screen-eligible (2AP, HA<=14, 0Ar, RotB<=2): {screen2:,}')

    db_size = os.path.getsize(DST_DB)
    print(f'  DB size: {db_size/1e9:.2f} GB')


if __name__ == '__main__':
    main()
