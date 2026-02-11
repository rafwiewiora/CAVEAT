"""CLI entry points for CAVEAT."""

from __future__ import annotations

import sys
import time

import click
from rdkit import Chem

from caveat.database import FragmentDatabase
from caveat.fragment import BRICSFragmenter
from caveat.query import find_replacements


@click.group()
def cli():
    """CAVEAT: Fragment-based molecular replacement tool."""
    pass


@cli.command()
@click.option("--input", "input_file", required=True, type=click.Path(exists=True),
              help="Input SMILES file (.smi)")
@click.option("--db", "db_path", required=True, type=click.Path(),
              help="Output database path")
@click.option("--method", default="brics", type=click.Choice(["brics"]),
              help="Fragmentation method")
@click.option("--n-confs", default=10, type=int,
              help="Number of conformers per fragment")
@click.option("--min-heavy", default=3, type=int,
              help="Minimum heavy atoms per fragment")
@click.option("--workers", default=1, type=int,
              help="Number of parallel workers for embedding (default: 1)")
def build(input_file, db_path, method, n_confs, min_heavy, workers):
    """Build a fragment database from a SMILES file."""
    click.echo(f"Reading molecules from {input_file}...")
    molecules = _read_smiles_file(input_file)
    click.echo(f"Read {len(molecules)} molecules")

    if not molecules:
        click.echo("No valid molecules found.", err=True)
        sys.exit(1)

    fragmenter = BRICSFragmenter(min_heavy_atoms=min_heavy)

    if workers > 1:
        click.echo(f"Building database at {db_path} with {workers} workers...")
    else:
        click.echo(f"Building database at {db_path}...")
    start = time.time()

    with FragmentDatabase(db_path) as db:
        last_msg = [None]

        def progress(current, total, phase=None):
            if current % 10 == 0 or current == total:
                if phase:
                    msg = f"  [{phase}] {current}/{total}..."
                else:
                    msg = f"  Processed {current}/{total} molecules..."
                if msg != last_msg[0]:
                    click.echo(msg)
                    last_msg[0] = msg

        stats = db.build(molecules, fragmenter=fragmenter, n_confs=n_confs,
                        progress_callback=progress, workers=workers)

    elapsed = time.time() - start
    click.echo(f"\nBuild complete in {elapsed:.1f}s:")
    click.echo(f"  Molecules processed: {stats['molecules']}")
    click.echo(f"  Unique fragments: {stats['fragments']}")
    click.echo(f"  Conformers generated: {stats['conformers']}")
    click.echo(f"  Skipped (invalid): {stats['skipped']}")


@cli.command()
@click.option("--db", "db_path", required=True, type=click.Path(exists=True),
              help="Fragment database path")
@click.option("--mol", "mol_smiles", required=True, type=str,
              help="Query molecule SMILES")
@click.option("--replace", "query_smarts", required=True, type=str,
              help="SMARTS pattern of substructure to replace")
@click.option("--top", "top_k", default=20, type=int,
              help="Number of top results to return")
@click.option("--tol", "tolerance", default=1.0, type=float,
              help="Geometric tolerance multiplier")
@click.option("--n-confs", default=10, type=int,
              help="Number of conformers for query molecule")
def query(db_path, mol_smiles, query_smarts, top_k, tolerance, n_confs):
    """Query the database for replacement fragments."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        click.echo(f"Invalid molecule SMILES: {mol_smiles}", err=True)
        sys.exit(1)

    click.echo(f"Querying database for replacements of '{query_smarts}'...")

    with FragmentDatabase(db_path) as db:
        try:
            results = find_replacements(
                mol, query_smarts, db,
                n_confs=n_confs, top_k=top_k, tolerance=tolerance,
                mol_smiles=mol_smiles,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    if not results:
        click.echo("No matching replacements found.")
        return

    click.echo(f"\nTop {len(results)} replacement fragments:")
    click.echo(f"{'Rank':<6}{'SMILES':<50}{'Score':<10}{'#APs':<6}{'#Heavy':<8}{'Sources':<8}")
    click.echo("-" * 88)
    for i, r in enumerate(results, 1):
        nha = r.num_heavy_atoms if r.num_heavy_atoms is not None else "?"
        sc = r.source_count if r.source_count is not None else "?"
        click.echo(f"{i:<6}{r.smiles:<50}{r.geometric_distance:<10.3f}{r.num_attachment_points:<6}{nha:<8}{sc:<8}")


@cli.command()
@click.option("--db", "db_path", required=True, type=click.Path(exists=True),
              help="Fragment database path")
def info(db_path):
    """Show database statistics."""
    with FragmentDatabase(db_path) as db:
        stats = db.get_stats()

    click.echo(f"Database: {db_path}")
    click.echo(f"  Fragments: {stats['num_fragments']}")
    click.echo(f"  Conformers: {stats['num_conformers']}")
    click.echo(f"  Vector pairs: {stats['num_vector_pairs']}")
    click.echo(f"  Source entries: {stats['num_sources']}")
    click.echo(f"  Unique source molecules: {stats['num_unique_sources']}")

    if "min_heavy_atoms" in stats:
        click.echo(f"  Heavy atoms: min={stats['min_heavy_atoms']}, "
                   f"max={stats['max_heavy_atoms']}, "
                   f"avg={stats['avg_heavy_atoms']}")

    if stats.get("ap_distribution"):
        click.echo("  Attachment point distribution:")
        for nap, count in sorted(stats["ap_distribution"].items()):
            click.echo(f"    {nap} APs: {count} fragments")


def _read_smiles_file(path: str) -> list[tuple[str, Chem.Mol]]:
    """Read a SMILES file and return (smiles, mol) pairs."""
    molecules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                molecules.append((smi, mol))
    return molecules


if __name__ == "__main__":
    cli()
