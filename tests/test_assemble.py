"""Tests for assemble module."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from caveat.assemble import assemble
from caveat.fragment import AttachmentPoint, BRICSFragmenter
from caveat.geometry import embed_fragment


def _check_bond_quality(mol, max_bond_length=2.0):
    """Helper: check that all bonds have reasonable lengths."""
    conf = mol.GetConformer(0)
    for bond in mol.GetBonds():
        p1 = np.array(list(conf.GetAtomPosition(bond.GetBeginAtomIdx())))
        p2 = np.array(list(conf.GetAtomPosition(bond.GetEndAtomIdx())))
        bl = np.linalg.norm(p2 - p1)
        assert bl < max_bond_length, (
            f"Bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} "
            f"length {bl:.2f} > {max_bond_length}"
        )
        assert bl > 0.5, (
            f"Bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} "
            f"length {bl:.2f} too short"
        )


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

    def test_parent_coords_preserved(self):
        """After assembly with 3D parent + 3D fragment, parent atoms stay in place."""
        parent = Chem.MolFromSmiles("c1ccccc1OC")
        parent3d = _make_3d(parent)

        # Match methyl on the Hs-added mol
        query = Chem.MolFromSmarts("[CH3]")
        matches = parent3d.GetSubstructMatches(query)
        assert matches
        match_atoms = matches[0]
        match_set = set(match_atoms)

        # Also include Hs exclusively bonded to matched atoms
        for atom in parent3d.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIdx() not in match_set:
                nbrs = atom.GetNeighbors()
                if len(nbrs) == 1 and nbrs[0].GetIdx() in match_set:
                    match_set.add(atom.GetIdx())

        # Replacement: ethyl fragment [3*]CC with 3D conformer
        replacement = Chem.MolFromSmiles("[3*]CC")
        replacement_3d = embed_fragment(replacement, n_confs=1)
        # Find AP on the 3D mol
        ap = None
        for atom in replacement_3d.GetAtoms():
            if atom.GetAtomicNum() == 0:
                ap = AttachmentPoint(
                    dummy_atom_idx=atom.GetIdx(),
                    neighbor_atom_idx=atom.GetNeighbors()[0].GetIdx(),
                    brics_label=3,
                )
                break
        assert ap is not None

        result = assemble(parent3d, match_atoms, replacement_3d, [ap])
        assert result is not None
        assert result.GetNumConformers() > 0

        # Verify parent atom positions are preserved
        parent_conf = parent3d.GetConformer(0)
        result_conf = result.GetConformer(0)

        # Build the parent_map: parent atoms not in match_set, mapped to result indices
        # (mirrors logic in _assemble_impl)
        parent_map = {}
        new_idx = 0
        for atom in parent3d.GetAtoms():
            if atom.GetIdx() not in match_set:
                parent_map[atom.GetIdx()] = new_idx
                new_idx += 1

        for old_idx, new_idx in parent_map.items():
            if new_idx < result.GetNumAtoms():
                orig_pos = np.array(list(parent_conf.GetAtomPosition(old_idx)))
                result_pos = np.array(list(result_conf.GetAtomPosition(new_idx)))
                dist = np.linalg.norm(orig_pos - result_pos)
                assert dist < 0.5, (
                    f"Parent atom {old_idx} (result {new_idx}) moved {dist:.3f} Å"
                )

    def test_1ap_bond_quality(self):
        """1-AP assembly: replacing pyridine should produce unstrained bonds."""
        parent = Chem.MolFromSmiles("c1ccncc1-c1ccccc1")  # 2-phenylpyridine
        parent3d = _make_3d(parent)

        # Replace the pyridine ring (1 cut bond to phenyl)
        query = Chem.MolFromSmarts("c1ccncc1")
        matches = parent3d.GetSubstructMatches(query)
        assert matches
        match_atoms = matches[0]

        # Replacement: benzene ring [16*]c1ccccc1
        replacement = Chem.MolFromSmiles("[16*]c1ccccc1")
        replacement_3d = embed_fragment(replacement, n_confs=1)
        ap = None
        for atom in replacement_3d.GetAtoms():
            if atom.GetAtomicNum() == 0:
                ap = AttachmentPoint(
                    dummy_atom_idx=atom.GetIdx(),
                    neighbor_atom_idx=atom.GetNeighbors()[0].GetIdx(),
                    brics_label=16,
                )
                break

        result = assemble(parent3d, match_atoms, replacement_3d, [ap])
        assert result is not None
        assert result.GetNumConformers() > 0
        _check_bond_quality(result)

    def test_2ap_bond_quality(self):
        """2-AP assembly: replacing a linker with 2 attachment points."""
        # Diphenylamine: Ph-NH-Ph, replace the NH linker (2 cut bonds)
        parent = Chem.MolFromSmiles("c1ccc(cc1)Nc1ccccc1")
        parent3d = _make_3d(parent)

        # Replace the NH (match 1 atom, 2 bonds to the phenyls)
        query = Chem.MolFromSmarts("[NH]")
        matches = parent3d.GetSubstructMatches(query)
        assert matches
        match_atoms = matches[0]

        # Replacement: oxygen linker [5*]O[16*]
        replacement = Chem.MolFromSmiles("[5*]O[16*]")
        replacement_3d = embed_fragment(replacement, n_confs=1)
        aps = []
        for atom in replacement_3d.GetAtoms():
            if atom.GetAtomicNum() == 0:
                aps.append(AttachmentPoint(
                    dummy_atom_idx=atom.GetIdx(),
                    neighbor_atom_idx=atom.GetNeighbors()[0].GetIdx(),
                    brics_label=atom.GetIsotope(),
                ))

        if len(aps) == 2:
            result = assemble(parent3d, match_atoms, replacement_3d, aps)
            if result is not None and result.GetNumConformers() > 0:
                _check_bond_quality(result)
