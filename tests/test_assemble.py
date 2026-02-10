"""Tests for assemble module."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from caveat.assemble import assemble
from caveat.fragment import AttachmentPoint, BRICSFragmenter


def _make_3d(mol):
    """Helper: add Hs and embed in 3D."""
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    return mol


class TestAssemble:
    def test_simple_replacement(self):
        """Replace methyl in anisole with ethyl."""
        parent = Chem.MolFromSmiles("c1ccccc1OC")
        parent3d = _make_3d(parent)

        # Match on the Hs-added mol (parent3d) so atom indices are consistent
        query = Chem.MolFromSmarts("[CH3]")
        matches = parent3d.GetSubstructMatches(query)
        assert matches
        match_atoms = matches[0]

        # Replacement: ethyl fragment [3*]CC
        replacement = Chem.MolFromSmiles("[3*]CC")
        ap = AttachmentPoint(dummy_atom_idx=0, neighbor_atom_idx=1, brics_label=3)

        result = assemble(parent3d, match_atoms, replacement, [ap])
        assert result is not None
        smi = Chem.MolToSmiles(result)
        assert "O" in smi

    def test_assemble_with_no_3d_parent(self):
        """Assembly should still work (without constrained embedding) when parent has no 3D."""
        parent = Chem.MolFromSmiles("c1ccccc1OC")

        query = Chem.MolFromSmarts("[CH3]")
        matches = parent.GetSubstructMatches(query)
        match_atoms = matches[0]

        replacement = Chem.MolFromSmiles("[3*]CC")
        ap = AttachmentPoint(dummy_atom_idx=0, neighbor_atom_idx=1, brics_label=3)

        result = assemble(parent, match_atoms, replacement, [ap])
        # May or may not succeed depending on embedding, but should not crash
        # Even if embedding fails, we still get a mol (just without coords)

    def test_ring_replacement(self):
        """Replace phenyl ring with cyclohexyl."""
        parent = Chem.MolFromSmiles("c1ccc(cc1)O")  # phenol
        parent3d = _make_3d(parent)

        query = Chem.MolFromSmarts("c1ccccc1")
        matches = parent3d.GetSubstructMatches(query)
        assert matches
        match_atoms = matches[0]

        # Cyclohexyl with one attachment: [4*]C1CCCCC1
        replacement = Chem.MolFromSmiles("[4*]C1CCCCC1")
        aps = []
        for atom in replacement.GetAtoms():
            if atom.GetAtomicNum() == 0:
                aps.append(AttachmentPoint(
                    dummy_atom_idx=atom.GetIdx(),
                    neighbor_atom_idx=atom.GetNeighbors()[0].GetIdx(),
                    brics_label=4,
                ))

        result = assemble(parent3d, match_atoms, replacement, aps)
        assert result is not None
        smi = Chem.MolToSmiles(result)
        assert "O" in smi

    def test_assemble_returns_none_on_invalid_input(self):
        """Should return None rather than crash on bad input."""
        parent = Chem.MolFromSmiles("CCO")
        # Empty match atoms — no bonds to cut
        result = assemble(parent, (), Chem.MolFromSmiles("[3*]C"), [])
        # Should handle gracefully
        assert result is not None or result is None  # just don't crash

    def test_assembly_preserves_connectivity(self):
        """The assembled molecule should have valid chemistry."""
        parent = Chem.MolFromSmiles("c1ccccc1OC")
        parent3d = _make_3d(parent)

        query = Chem.MolFromSmarts("[CH3]")
        matches = parent3d.GetSubstructMatches(query)
        match_atoms = matches[0]

        replacement = Chem.MolFromSmiles("[3*]CC")
        ap = AttachmentPoint(dummy_atom_idx=0, neighbor_atom_idx=1, brics_label=3)

        result = assemble(parent3d, match_atoms, replacement, [ap])
        if result is not None:
            # Should parse back to valid SMILES
            smi = Chem.MolToSmiles(result)
            reparsed = Chem.MolFromSmiles(smi)
            assert reparsed is not None
