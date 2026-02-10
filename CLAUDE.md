# CAVEAT — Fragment-Based Molecular Replacement Tool

## Overview
Open-source tool inspired by CAVEAT (Lauri & Bartlett, 1994) for bioisosteric fragment replacement. Pipeline: fragment molecules via BRICS, build indexed database keyed by 3D attachment vector geometry, query for compatible replacement fragments.

## Tech Stack
- Python 3.11, RDKit, NumPy, SciPy, Click, SQLite
- Conda (conda-forge) for environment management
- pytest for testing

## Setup
```bash
conda activate caveat
pip install -e ".[dev]"
```

## Project Structure
- `caveat/fragment.py` — BRICS fragmentation engine (Fragmenter base class, BRICSFragmenter)
- `caveat/geometry.py` — 3D bond vector computation, CAVEAT geometric descriptors
- `caveat/database.py` — SQLite fragment database with KDTree indexing
- `caveat/query.py` — Query engine: find replacement fragments for a substructure
- `caveat/assemble.py` — Stitch replacement fragments into parent molecules
- `caveat/cli.py` — CLI: `caveat build`, `caveat query`, `caveat info`
- `data/chembl_sample.smi` — ~100 diverse drug-like test molecules

## Key Commands
```bash
# Run tests
pytest tests/ -v

# Build fragment database
caveat build --input data/chembl_sample.smi --db fragments.db --n-confs 10

# Query for replacements (e.g., replace N-methylpiperazine in imatinib)
caveat query --db fragments.db --mol "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5" --replace "CN1CCNCC1" --top 10

# Database stats
caveat info --db fragments.db
```

## Architecture Notes
- Geometric descriptor: (distance, angle1, angle2, dihedral) between pairs of attachment point vectors — following original CAVEAT paper parameterization
- KDTree in 4D normalized space for fast nearest-neighbor search
- Fragments deduplicated by canonical SMILES; source molecules tracked separately
- Single attachment point fragments handled via simple enumeration (no geometric pair to compare)
