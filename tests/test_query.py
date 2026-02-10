"""Tests for query module."""

import pytest
from rdkit import Chem

from caveat.query import find_replacements, _find_cut_bonds, ReplacementResult


class TestFindCutBonds:
    def test_single_cut_bond(self):
        mol = Chem.MolFromSmiles("c1ccccc1OC")  # phenol methyl ether
        # Match the methyl (C bonded to O)
        query = Chem.MolFromSmarts("[CH3]")
        matches = mol.GetSubstructMatches(query)
        assert matches
        match_atoms = set(matches[0])
        cuts = _find_cut_bonds(mol, match_atoms)
        assert len(cuts) == 1

    def test_two_cut_bonds(self):
        mol = Chem.MolFromSmiles("c1ccccc1Nc1ccccc1")  # diphenylamine
        # Match the NH linker
        query = Chem.MolFromSmarts("[NH]")
        matches = mol.GetSubstructMatches(query)
        assert matches
        match_atoms = set(matches[0])
        cuts = _find_cut_bonds(mol, match_atoms)
        assert len(cuts) == 2


class TestFindReplacements:
    def test_single_ap_query(self, small_db):
        """Query with a single attachment point substructure."""
        # Use a molecule with a clear single-AP substructure
        mol = Chem.MolFromSmiles("c1ccccc1OC")  # anisole
        results = find_replacements(
            mol, "[CH3]", small_db, n_confs=2, top_k=5,
        )
        # Should return some 1-AP fragments
        assert isinstance(results, list)

    def test_multi_ap_query(self, small_db):
        """Query with a 2-AP substructure."""
        stats = small_db.get_stats()
        if stats["num_vector_pairs"] == 0:
            pytest.skip("No vector pairs in test DB")

        mol = Chem.MolFromSmiles("c1ccccc1C(=O)Nc1ccccc1")  # benzanilide
        results = find_replacements(
            mol, "C(=O)N", small_db, n_confs=2, top_k=10,
        )
        assert isinstance(results, list)

    def test_invalid_smarts(self, small_db):
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            find_replacements(mol, "[invalid", small_db)

    def test_no_match(self, small_db):
        mol = Chem.MolFromSmiles("CCO")  # ethanol
        with pytest.raises(ValueError, match="not found"):
            find_replacements(mol, "c1ccc2ccccc2c1", small_db)  # naphthalene

    def test_results_sorted(self, small_db):
        """Results should be sorted by geometric_distance."""
        stats = small_db.get_stats()
        if stats["num_vector_pairs"] == 0:
            pytest.skip("No vector pairs in test DB")

        mol = Chem.MolFromSmiles("c1ccccc1C(=O)Nc1ccccc1")
        results = find_replacements(
            mol, "C(=O)N", small_db, n_confs=2, top_k=20,
        )
        for i in range(len(results) - 1):
            assert results[i].geometric_distance <= results[i + 1].geometric_distance

    def test_replacement_result_fields(self, small_db):
        mol = Chem.MolFromSmiles("c1ccccc1OC")
        results = find_replacements(mol, "[CH3]", small_db, n_confs=2, top_k=3)
        for r in results:
            assert isinstance(r, ReplacementResult)
            assert r.smiles
            assert r.fragment_id > 0
            assert r.num_attachment_points >= 1

    def test_sql_query_mode(self, small_db):
        """Test with use_kdtree=False."""
        stats = small_db.get_stats()
        if stats["num_vector_pairs"] == 0:
            pytest.skip("No vector pairs in test DB")

        mol = Chem.MolFromSmiles("c1ccccc1C(=O)Nc1ccccc1")
        results = find_replacements(
            mol, "C(=O)N", small_db,
            n_confs=2, top_k=10, use_kdtree=False,
        )
        assert isinstance(results, list)
