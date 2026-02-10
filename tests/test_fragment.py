"""Tests for fragment module."""

import pytest
from rdkit import Chem

from caveat.fragment import (
    AttachmentPoint,
    Fragment,
    BRICSFragmenter,
)


class TestAttachmentPoint:
    def test_to_dict_roundtrip(self):
        ap = AttachmentPoint(
            dummy_atom_idx=0, neighbor_atom_idx=1,
            brics_label=3, bond_order=1,
        )
        d = ap.to_dict()
        ap2 = AttachmentPoint.from_dict(d)
        assert ap == ap2

    def test_dict_keys(self):
        ap = AttachmentPoint(0, 1, 3, 1)
        d = ap.to_dict()
        assert set(d.keys()) == {"dummy_atom_idx", "neighbor_atom_idx", "brics_label", "bond_order"}


class TestFragment:
    def test_num_heavy_atoms(self):
        mol = Chem.MolFromSmiles("[3*]c1ccccc1")
        frag = Fragment(
            mol=mol,
            smiles="[3*]c1ccccc1",
            attachment_points=[
                AttachmentPoint(0, 1, 3, 1),
            ],
        )
        # 6 carbons = 6 heavy atoms (dummy not counted)
        assert frag.num_heavy_atoms == 6

    def test_num_attachment_points(self):
        mol = Chem.MolFromSmiles("[3*]CC[5*]")
        frag = Fragment(
            mol=mol,
            smiles="[3*]CC[5*]",
            attachment_points=[
                AttachmentPoint(0, 1, 3, 1),
                AttachmentPoint(3, 2, 5, 1),
            ],
        )
        assert frag.num_attachment_points == 2


class TestBRICSFragmenter:
    def test_fragment_aspirin(self, aspirin_mol):
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(aspirin_mol)
        assert len(frags) > 0
        for frag in frags:
            assert frag.num_heavy_atoms >= 3
            assert len(frag.attachment_points) > 0

    def test_fragment_ibuprofen(self):
        mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(mol)
        assert len(frags) > 0

    def test_fragment_celecoxib(self, celecoxib_mol):
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(celecoxib_mol)
        assert len(frags) > 0

    def test_fragment_none(self):
        f = BRICSFragmenter()
        result = f.fragment(None)
        assert result == []

    def test_fragment_no_brics_bonds(self):
        # Simple molecule with no BRICS bonds (e.g., methane)
        mol = Chem.MolFromSmiles("C")
        f = BRICSFragmenter()
        result = f.fragment(mol)
        assert result == []

    def test_fragments_have_smiles(self, aspirin_mol):
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(aspirin_mol)
        for frag in frags:
            assert frag.smiles
            # Should be parseable
            assert Chem.MolFromSmiles(frag.smiles) is not None

    def test_fragments_have_brics_labels(self, aspirin_mol):
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(aspirin_mol)
        for frag in frags:
            assert isinstance(frag.brics_labels, list)
            for label in frag.brics_labels:
                assert isinstance(label, int)

    def test_source_smiles_preserved(self):
        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(mol, source_smiles="aspirin")
        for frag in frags:
            assert frag.source_smiles == "aspirin"

    def test_min_heavy_atoms_filter(self, aspirin_mol):
        f_strict = BRICSFragmenter(min_heavy_atoms=5)
        f_loose = BRICSFragmenter(min_heavy_atoms=1)
        frags_strict = f_strict.fragment(aspirin_mol)
        frags_loose = f_loose.fragment(aspirin_mol)
        assert len(frags_strict) <= len(frags_loose)

    def test_dummy_atom_connectivity(self, aspirin_mol):
        f = BRICSFragmenter()
        frags = f.fragment(aspirin_mol)
        for frag in frags:
            for ap in frag.attachment_points:
                atom = frag.mol.GetAtomWithIdx(ap.dummy_atom_idx)
                assert atom.GetAtomicNum() == 0  # is dummy
                neighbor = frag.mol.GetAtomWithIdx(ap.neighbor_atom_idx)
                assert neighbor.GetAtomicNum() > 0  # is real atom
                # Verify they are bonded
                bond = frag.mol.GetBondBetweenAtoms(ap.dummy_atom_idx, ap.neighbor_atom_idx)
                assert bond is not None

    def test_imatinib_produces_many_fragments(self, imatinib_mol):
        f = BRICSFragmenter(min_heavy_atoms=3)
        frags = f.fragment(imatinib_mol)
        # Imatinib is a large molecule with many BRICS bonds
        assert len(frags) >= 3
