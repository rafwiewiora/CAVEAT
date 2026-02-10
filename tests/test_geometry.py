"""Tests for geometry module."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from caveat.geometry import (
    embed_fragment,
    get_exit_vector,
    compute_vector_pair_descriptor,
    align_single_vector,
    align_two_vectors,
    apply_transform,
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


class TestApplyTransform:
    def test_identity_preserves_positions(self):
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = apply_transform(positions, np.eye(4))
        np.testing.assert_allclose(result, positions, atol=1e-10)

    def test_translation(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        T = np.eye(4)
        T[:3, 3] = [10.0, 20.0, 30.0]
        result = apply_transform(positions, T)
        expected = np.array([[10.0, 20.0, 30.0], [11.0, 20.0, 30.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rotation_90_z(self):
        positions = np.array([[1.0, 0.0, 0.0]])
        R = np.eye(4)
        R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        result = apply_transform(positions, R)
        expected = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestAlignSingleVector:
    def test_translation_only(self):
        """When vectors already point the same way, only translation needed."""
        frag_neighbor = np.array([0.0, 0.0, 0.0])
        frag_dummy = np.array([1.0, 0.0, 0.0])
        target_pos = np.array([5.0, 5.0, 5.0])
        target_dir = np.array([6.0, 5.0, 5.0])

        frag_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        transform = align_single_vector(frag_positions, frag_neighbor, frag_dummy, target_pos, target_dir)
        result = apply_transform(frag_positions, transform)

        # Neighbor should land at target_pos
        np.testing.assert_allclose(result[0], target_pos, atol=1e-6)
        # Dummy should be along the target direction from neighbor
        moved_dir = result[1] - result[0]
        expected_dir = target_dir - target_pos
        moved_dir /= np.linalg.norm(moved_dir)
        expected_dir /= np.linalg.norm(expected_dir)
        np.testing.assert_allclose(moved_dir, expected_dir, atol=1e-6)

    def test_rotation_and_translation(self):
        """Fragment exit vector along X, target along Y."""
        frag_neighbor = np.array([0.0, 0.0, 0.0])
        frag_dummy = np.array([1.0, 0.0, 0.0])
        target_pos = np.array([3.0, 3.0, 3.0])
        target_dir = np.array([3.0, 4.0, 3.0])  # Y direction

        frag_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        transform = align_single_vector(frag_positions, frag_neighbor, frag_dummy, target_pos, target_dir)
        result = apply_transform(frag_positions, transform)

        # Neighbor at target
        np.testing.assert_allclose(result[0], target_pos, atol=1e-6)
        # Exit vector direction aligned
        moved_dir = result[1] - result[0]
        expected_dir = target_dir - target_pos
        moved_dir /= np.linalg.norm(moved_dir)
        expected_dir /= np.linalg.norm(expected_dir)
        np.testing.assert_allclose(moved_dir, expected_dir, atol=1e-6)

    def test_opposite_vectors(self):
        """Fragment exit vector opposite to target direction."""
        frag_neighbor = np.array([0.0, 0.0, 0.0])
        frag_dummy = np.array([1.0, 0.0, 0.0])
        target_pos = np.array([0.0, 0.0, 0.0])
        target_dir = np.array([-1.0, 0.0, 0.0])

        frag_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        transform = align_single_vector(frag_positions, frag_neighbor, frag_dummy, target_pos, target_dir)
        result = apply_transform(frag_positions, transform)

        np.testing.assert_allclose(result[0], target_pos, atol=1e-6)
        moved_dir = result[1] - result[0]
        expected_dir = np.array([-1.0, 0.0, 0.0])
        moved_dir /= np.linalg.norm(moved_dir)
        np.testing.assert_allclose(moved_dir, expected_dir, atol=1e-6)


class TestAlignTwoVectors:
    def test_identity_case(self):
        """Source and target are the same — transform should be identity-like."""
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ])
        transform = align_two_vectors(pts, pts)
        result = apply_transform(pts, transform)
        np.testing.assert_allclose(result, pts, atol=1e-6)

    def test_pure_translation(self):
        """Points shifted by a constant offset."""
        source = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        offset = np.array([5.0, 10.0, 15.0])
        target = source + offset
        transform = align_two_vectors(source, target)
        result = apply_transform(source, transform)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_rotation_90_z(self):
        """90° rotation around Z axis."""
        source = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ])
        # Rotate 90° around Z: (x,y) -> (-y, x)
        target = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 2.0, 0.0],
        ])
        transform = align_two_vectors(source, target)
        result = apply_transform(source, transform)
        np.testing.assert_allclose(result, target, atol=1e-5)


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
