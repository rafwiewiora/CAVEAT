"""Tests for database module."""

import json
import os
import tempfile

import pytest
from rdkit import Chem

from caveat.database import FragmentDatabase
from caveat.fragment import BRICSFragmenter


class TestFragmentDatabase:
    def test_create_database(self, tmp_db_path):
        db = FragmentDatabase(tmp_db_path)
        assert os.path.exists(tmp_db_path)
        db.close()

    def test_build_from_molecules(self, tmp_db_path, simple_molecules):
        db = FragmentDatabase(tmp_db_path)
        stats = db.build(simple_molecules, n_confs=2)
        assert stats["molecules"] > 0
        assert stats["fragments"] > 0
        assert stats["conformers"] > 0
        db.close()

    def test_get_stats(self, small_db):
        stats = small_db.get_stats()
        assert stats["num_fragments"] > 0
        assert stats["num_conformers"] > 0
        assert "ap_distribution" in stats

    def test_get_fragment(self, small_db):
        frag = small_db.get_fragment(1)
        assert frag is not None
        assert "smiles" in frag
        assert "num_heavy_atoms" in frag
        assert "attachment_points" in frag

    def test_get_fragment_not_found(self, small_db):
        frag = small_db.get_fragment(99999)
        assert frag is None

    def test_get_sources(self, small_db):
        # At least one fragment should have sources
        frag = small_db.get_fragment(1)
        if frag:
            sources = small_db.get_sources(1)
            assert len(sources) >= 1

    def test_deduplicate_fragments(self, tmp_db_path):
        """Building with the same molecule twice should not create duplicates."""
        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
        molecules = [
            ("aspirin1", mol),
            ("aspirin2", mol),
        ]
        db = FragmentDatabase(tmp_db_path)
        stats = db.build(molecules, n_confs=2)
        db_stats = db.get_stats()
        # Fragments from both copies should be deduplicated
        # but source_count should reflect both
        assert db_stats["num_sources"] >= db_stats["num_fragments"]
        db.close()

    def test_context_manager(self, tmp_db_path):
        with FragmentDatabase(tmp_db_path) as db:
            stats = db.get_stats()
            assert stats["num_fragments"] == 0

    def test_query_by_geometry(self, small_db):
        """Query with very wide tolerances should return results if there are vector pairs."""
        stats = small_db.get_stats()
        if stats["num_vector_pairs"] > 0:
            results = small_db.query_by_geometry(
                distance=3.0, angle1=120.0, angle2=120.0, dihedral=0.0,
                tolerances={
                    "distance": 100.0, "angle1": 180.0,
                    "angle2": 180.0, "dihedral": 360.0,
                },
            )
            assert len(results) > 0

    def test_query_by_kdtree(self, small_db):
        stats = small_db.get_stats()
        if stats["num_vector_pairs"] > 0:
            results = small_db.query_by_kdtree(
                distance=3.0, angle1=120.0, angle2=120.0, dihedral=0.0,
                k=5,
            )
            assert len(results) > 0

    def test_build_progress_callback(self, tmp_db_path, simple_molecules):
        calls = []
        db = FragmentDatabase(tmp_db_path)
        db.build(
            simple_molecules, n_confs=2,
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        assert len(calls) > 0
        assert calls[-1][0] == len(simple_molecules)
        db.close()

    def test_empty_molecules_list(self, tmp_db_path):
        db = FragmentDatabase(tmp_db_path)
        stats = db.build([], n_confs=2)
        assert stats["fragments"] == 0
        assert stats["molecules"] == 0
        db.close()

    def test_none_molecules_skipped(self, tmp_db_path):
        molecules = [("bad", None), ("good", Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))]
        db = FragmentDatabase(tmp_db_path)
        stats = db.build(molecules, n_confs=2)
        assert stats["skipped"] == 1
        db.close()

    def test_parallel_build(self, tmp_db_path, simple_molecules):
        """Parallel build should produce the same results as serial."""
        import tempfile

        # Build serial
        db_serial = FragmentDatabase(tmp_db_path)
        stats_serial = db_serial.build(simple_molecules, n_confs=2, workers=1)

        # Build parallel
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            parallel_path = f.name
        db_parallel = FragmentDatabase(parallel_path)
        stats_parallel = db_parallel.build(simple_molecules, n_confs=2, workers=2)

        assert stats_serial["fragments"] == stats_parallel["fragments"]
        assert stats_serial["molecules"] == stats_parallel["molecules"]

        serial_stats = db_serial.get_stats()
        parallel_stats = db_parallel.get_stats()
        assert serial_stats["num_fragments"] == parallel_stats["num_fragments"]
        assert serial_stats["num_conformers"] == parallel_stats["num_conformers"]
        assert serial_stats["num_vector_pairs"] == parallel_stats["num_vector_pairs"]

        db_serial.close()
        db_parallel.close()
        os.unlink(parallel_path)
