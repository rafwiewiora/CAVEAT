"""Shared fixtures for CAVEAT tests."""

import os
import tempfile

import pytest
from rdkit import Chem

from caveat.database import FragmentDatabase
from caveat.fragment import BRICSFragmenter


# Well-known drug SMILES for testing
IMATINIB = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CELECOXIB = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
IBUPROFEN = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
PIPERAZINE_SMARTS = "CN1CCNCC1"  # N-methylpiperazine in imatinib


@pytest.fixture
def imatinib_mol():
    return Chem.MolFromSmiles(IMATINIB)


@pytest.fixture
def aspirin_mol():
    return Chem.MolFromSmiles(ASPIRIN)


@pytest.fixture
def caffeine_mol():
    return Chem.MolFromSmiles(CAFFEINE)


@pytest.fixture
def celecoxib_mol():
    return Chem.MolFromSmiles(CELECOXIB)


@pytest.fixture
def simple_molecules():
    """A small set of molecules for quick tests."""
    smiles_list = [
        ASPIRIN,
        IBUPROFEN,
        CELECOXIB,
        "CC(=O)NC1=CC=C(C=C1)O",  # acetaminophen
        "C1=CC=C(C=C1)C(=O)O",  # benzoic acid
        "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2N",  # toluamide
        "C1=CC=C(C=C1)C2=CC=CC=N2",  # 2-phenylpyridine
        "C1=CC=C(C=C1)OC2=CC=CC=C2",  # diphenyl ether
    ]
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append((smi, mol))
    return mols


@pytest.fixture
def brics_fragmenter():
    return BRICSFragmenter(min_heavy_atoms=3)


@pytest.fixture
def tmp_db_path():
    """Provide a temporary database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def small_db(tmp_db_path, simple_molecules):
    """Build a small database from test molecules."""
    db = FragmentDatabase(tmp_db_path)
    db.build(simple_molecules, n_confs=3)
    yield db
    db.close()
