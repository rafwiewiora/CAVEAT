"""Tests for geometry module."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from caveat.geometry import (
    embed_fragment,
    get_exit_vector,
    compute_vector_pair_descriptor,
    _angle,
    _dihedral,
    canonicalize_dihedral,
)


class TestAngle:
    def test_right_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        assert abs(_angle(a, b, c) - 90.0) < 0.1

    def test_straight_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        assert abs(_angle(a, b, c) - 180.0) < 0.1

    def test_zero_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        assert abs(_angle(a, b, c) - 0.0) < 0.1

    def test_45_degree(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        assert abs(_angle(a, b, c) - 45.0) < 0.1


class TestDihedral:
    def test_cis_dihedral(self):
        # cis configuration -> dihedral ~ 0
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 0.0])
        d = np.array([1.0, 0.0, 0.0])
        dih = _dihedral(a, b, c, d)
        assert abs(dih) < 1.0 or abs(abs(dih) - 360) < 1.0

    def test_trans_dihedral(self):
        # trans configuration -> dihedral ~ 180 or -180
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 0.0])
        d = np.array([-1.0, 0.0, 0.0])
        dih = _dihedral(a, b, c, d)
        assert abs(abs(dih) - 180.0) < 1.0

    def test_90_degree_dihedral(self):
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 0.0])
        d = np.array([0.0, 0.0, 1.0])
        dih = _dihedral(a, b, c, d)
        assert abs(abs(dih) - 90.0) < 1.0


class TestCanonicalDihedral:
    def test_positive(self):
        assert canonicalize_dihedral(45.0) == 45.0

    def test_negative(self):
        assert canonicalize_dihedral(-45.0) == 45.0


class TestEmbedFragment:
    def test_embed_simple_fragment(self):
        # A simple fragment with one dummy atom: [3*]c1ccccc1
        mol = Chem.MolFromSmiles("[3*]c1ccccc1")
        assert mol is not None
        result = embed_fragment(mol, n_confs=3)
        assert result.GetNumConformers() >= 1

    def test_embed_two_attachment(self):
        # Fragment with two dummy atoms
        mol = Chem.MolFromSmiles("[3*]C(=O)N[5*]")
        assert mol is not None
        result = embed_fragment(mol, n_confs=3)
        assert result.GetNumConformers() >= 1

    def test_embed_ring_fragment(self):
        mol = Chem.MolFromSmiles("[4*]C1CCNCC1")
        assert mol is not None
        result = embed_fragment(mol, n_confs=5)
        assert result.GetNumConformers() >= 1


class TestGetExitVector:
    def test_exit_vector_direction(self):
        mol = Chem.MolFromSmiles("[3*]c1ccccc1")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        assert mol.GetNumConformers() > 0

        # Find the dummy atom
        dummy_idx = None
        neighbor_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx = atom.GetIdx()
                neighbor_idx = atom.GetNeighbors()[0].GetIdx()
                break

        assert dummy_idx is not None
        conf = mol.GetConformer(0)
        ev = get_exit_vector(conf, dummy_idx, neighbor_idx)

        # Direction should be a unit vector
        assert abs(np.linalg.norm(ev.direction) - 1.0) < 1e-5
        # Origin should be at the neighbor position
        pos = np.array(conf.GetAtomPosition(neighbor_idx))
        assert np.allclose(ev.origin, pos, atol=1e-5)


class TestComputeVectorPairDescriptor:
    def test_descriptor_values_reasonable(self):
        # Fragment with 2 dummy atoms: [3*]CC[5*]
        mol = Chem.MolFromSmiles("[3*]CC[5*]")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        assert mol.GetNumConformers() > 0

        # Find dummy atoms
        dummies = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbor = atom.GetNeighbors()[0]
                dummies.append((atom.GetIdx(), neighbor.GetIdx()))

        assert len(dummies) == 2
        conf = mol.GetConformer(0)
        desc = compute_vector_pair_descriptor(
            conf,
            dummies[0][0], dummies[0][1],
            dummies[1][0], dummies[1][1],
        )

        # Distance should be around 1.5 Å (C-C bond)
        assert 1.0 < desc.distance < 2.5
        # Angles should be reasonable (not 0 or 180 for sp3)
        assert 30 < desc.angle1 < 170
        assert 30 < desc.angle2 < 170
        # Dihedral in [-180, 180]
        assert -180 <= desc.dihedral <= 180

    def test_as_array(self):
        mol = Chem.MolFromSmiles("[3*]CC[5*]")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        dummies = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummies.append((atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()))

        conf = mol.GetConformer(0)
        desc = compute_vector_pair_descriptor(
            conf, dummies[0][0], dummies[0][1], dummies[1][0], dummies[1][1],
        )
        arr = desc.as_array()
        assert arr.shape == (4,)
        assert arr[0] == desc.distance
