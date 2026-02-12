#!/usr/bin/env python
"""Backfill num_rotatable_bonds and num_aromatic_rings in chembl_full_frags.db.

Uses multiprocessing to compute descriptors from SMILES, then batch-updates DB.
"""
import sqlite3
import multiprocessing as mp
import time
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

DB_PATH = '/Users/rafal/repos/CAVEAT/chembl_full_frags.db'
BATCH_SIZE = 50000
NUM_WORKERS = 14


def compute_descriptors(batch):
    """Compute (id, num_rot_bonds, num_arom_rings) for a batch of (id, smiles)."""
    results = []
    for frag_id, smi in batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append((frag_id, 0, 0))
            continue
        nrb = Descriptors.NumRotatableBonds(mol)
        nar = Descriptors.NumAromaticRings(mol)
        results.append((frag_id, nrb, nar))
    return results


def main():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute('SELECT COUNT(*) FROM fragments WHERE num_rotatable_bonds IS NULL').fetchone()[0]
    print(f'Total rows to backfill: {total:,}')

    if total == 0:
        print('Nothing to do.')
        conn.close()
        return

    # Read all (id, smiles) for NULL rows
    print('Reading SMILES from DB...')
    t0 = time.time()
    cursor = conn.execute(
        'SELECT id, canonical_smiles FROM fragments WHERE num_rotatable_bonds IS NULL'
    )

    pool = mp.Pool(NUM_WORKERS)
    processed = 0
    batch = []

    def flush_results(results_list):
        nonlocal processed
        conn.executemany(
            'UPDATE fragments SET num_rotatable_bonds=?, num_aromatic_rings=? WHERE id=?',
            [(nrb, nar, fid) for fid, nrb, nar in results_list]
        )
        processed += len(results_list)

    pending_futures = []

    for row in cursor:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            # Submit batch to pool
            pending_futures.append(pool.apply_async(compute_descriptors, (batch,)))
            batch = []

            # Drain completed futures to avoid memory buildup
            if len(pending_futures) >= NUM_WORKERS * 2:
                for fut in pending_futures:
                    flush_results(fut.get())
                conn.commit()
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f'  {processed:>12,} / {total:,}  ({processed*100/total:.1f}%)  '
                      f'{rate:.0f} rows/s  ETA: {eta/60:.1f} min', flush=True)
                pending_futures = []

    # Submit remaining batch
    if batch:
        pending_futures.append(pool.apply_async(compute_descriptors, (batch,)))

    # Drain remaining futures
    for fut in pending_futures:
        flush_results(fut.get())
    conn.commit()

    pool.close()
    pool.join()

    elapsed = time.time() - t0
    print(f'\nDone! Processed {processed:,} rows in {elapsed:.0f}s ({processed/elapsed:.0f} rows/s)')
    conn.close()


if __name__ == '__main__':
    main()
