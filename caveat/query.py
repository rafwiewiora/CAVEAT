"""Query engine for CAVEAT.

Given a molecule and a substructure to replace, find compatible replacement
fragments from the database ranked by geometric similarity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from caveat.database import FragmentDatabase
from caveat.geometry import (
    embed_fragment,
    compute_vector_pair_descriptor,
    ExitVector,
    VectorPairDescriptor,
)

logger = logging.getLogger(__name__)


@dataclass
class CutBond:
    """A bond between the matched substructure and the rest of the molecule."""
    matched_atom_idx: int  # atom in the matched substructure
    external_atom_idx: int  # atom outside the substructure
    bond_idx: int
    bond_order: int


@dataclass
class ReplacementResult:
    """A candidate replacement fragment."""
    fragment_id: int
    smiles: str
    geometric_distance: float
    num_attachment_points: int
    num_heavy_atoms: Optional[int] = None
    source_count: Optional[int] = None
    mol: Optional[Chem.Mol] = None


def find_replacements(
    mol: Chem.Mol,
    query_smarts: str,
    db: FragmentDatabase,
    n_confs: int = 10,
    top_k: int = 20,
    tolerance: float = 1.0,
    use_kdtree: bool = False,
    mol_smiles: Optional[str] = None,
    mol_3d: Optional[Chem.Mol] = None,
) -> list[ReplacementResult]:
    """Find replacement fragments for a substructure in a molecule.

    Args:
        mol: the query molecule (used for substructure matching)
        query_smarts: SMARTS pattern identifying the substructure to replace
        db: the fragment database
        n_confs: number of conformers for the query molecule
        top_k: maximum number of results to return
        tolerance: geometric tolerance multiplier (1.0 = default tolerances:
                   distance ±0.5Å, angles ±15°, dihedral ±30°)
        use_kdtree: use KDTree for fast ranked search (no hard cutoff) or
                    SQL range query with hard tolerance cutoffs (default)
        mol_smiles: optional SMILES for the query molecule
        mol_3d: optional pre-embedded 3D molecule (skips conformer generation)

    Returns:
        list of ReplacementResult, sorted by geometric_distance
    """
    if mol_smiles is None:
        mol_smiles = Chem.MolToSmiles(mol)

    # Step 1: Find the substructure match
    query_mol = Chem.MolFromSmarts(query_smarts)
    if query_mol is None:
        raise ValueError(f"Invalid SMARTS pattern: {query_smarts}")

    matches = mol.GetSubstructMatches(query_mol)
    if not matches:
        raise ValueError(f"Substructure {query_smarts} not found in molecule")

    match_atoms = set(matches[0])  # use first match

    # Step 2: Find cut bonds
    cut_bonds = _find_cut_bonds(mol, match_atoms)
    if not cut_bonds:
        raise ValueError("No bonds cross the substructure boundary — nothing to replace")

    num_aps = len(cut_bonds)
    logger.info(f"Found {num_aps} cut bond(s) for replacement")

    # Step 3: Get 3D molecule — use pre-embedded or generate conformers
    if mol_3d is not None:
        mol3d = mol_3d
        if mol3d.GetNumConformers() == 0:
            raise ValueError("Provided mol_3d has no conformers")
    else:
        mol3d = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        n_gen = AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, params=params)
        if n_gen == 0:
            params.useRandomCoords = True
            n_gen = AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, params=params)
        if n_gen == 0:
            raise ValueError("Could not generate 3D conformers for query molecule")

        try:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol3d, numThreads=0)
        except Exception:
            pass

    # Step 4: Compute exit vector geometry at cut points
    # We need to map the original atom indices to the Hs-added mol
    # RDKit AddHs preserves the original atom ordering at the beginning
    descriptors = []
    for conf_idx in range(mol3d.GetNumConformers()):
        conf = mol3d.GetConformer(conf_idx)

        if num_aps == 1:
            # Single attachment point — search by single exit vector
            cb = cut_bonds[0]
            ev = _compute_single_exit_vector(conf, cb, match_atoms)
            descriptors.append(("single", ev, conf_idx))

        elif num_aps >= 2:
            # Multiple attachment points — compute pairwise descriptors
            for i in range(len(cut_bonds)):
                for j in range(i + 1, len(cut_bonds)):
                    cb1, cb2 = cut_bonds[i], cut_bonds[j]
                    desc = _compute_cut_pair_descriptor(conf, cb1, cb2, match_atoms)
                    if desc is not None:
                        descriptors.append(("pair", desc, conf_idx))

    # Step 5: Search the database
    all_results = {}
    for desc_type, desc_data, conf_idx in descriptors:
        if desc_type == "single":
            # For single AP fragments, rank by BRICS label compatibility
            # and heavy atom count similarity (no geometric pair to compare)
            cb = cut_bonds[0]
            target_n_heavy = sum(
                1 for a in match_atoms
                if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
            )
            # Determine if the cut bond enters an aromatic system
            matched_is_aromatic = mol.GetAtomWithIdx(
                cb.matched_atom_idx
            ).GetIsAromatic()
            aromatic_labels = {7, 8, 9, 14, 15, 16}

            rows = db.conn.execute(
                """SELECT id, canonical_smiles, num_heavy_atoms, source_count,
                          brics_labels
                   FROM fragments WHERE num_attachment_points = 1""",
            ).fetchall()
            for row in rows:
                fid, smi, nha, sc, labels_json = row
                import json as _json
                labels = _json.loads(labels_json) if labels_json else []
                # Check if fragment's BRICS label matches the bond character
                frag_is_aromatic = any(l in aromatic_labels for l in labels)
                label_penalty = 0.0 if frag_is_aromatic == matched_is_aromatic else 5.0
                size_dist = abs(nha - target_n_heavy) / max(target_n_heavy, 1)
                geo_dist = label_penalty + size_dist
                if fid not in all_results or geo_dist < all_results[fid].geometric_distance:
                    all_results[fid] = ReplacementResult(
                        fragment_id=fid,
                        smiles=smi,
                        geometric_distance=geo_dist,
                        num_attachment_points=1,
                        num_heavy_atoms=nha,
                        source_count=sc,
                    )

        elif desc_type == "pair":
            vpd = desc_data
            if use_kdtree:
                hits = db.query_by_kdtree(
                    vpd.distance, vpd.angle1, vpd.angle2, vpd.dihedral,
                    k=top_k * 2,
                    num_attachment_points=num_aps,
                )
            else:
                tolerances = {
                    "distance": 0.5 * tolerance,
                    "angle1": 15.0 * tolerance,
                    "angle2": 15.0 * tolerance,
                    "dihedral": 30.0 * tolerance,
                }
                hits = db.query_by_geometry(
                    vpd.distance, vpd.angle1, vpd.angle2, vpd.dihedral,
                    tolerances=tolerances,
                    num_attachment_points=num_aps,
                )

            for hit in hits:
                fid = hit["fragment_id"]
                gdist = hit["geometric_distance"]
                if fid not in all_results or gdist < all_results[fid].geometric_distance:
                    frag_info = db.get_fragment(fid)
                    all_results[fid] = ReplacementResult(
                        fragment_id=fid,
                        smiles=hit["smiles"],
                        geometric_distance=gdist,
                        num_attachment_points=hit["num_attachment_points"],
                        num_heavy_atoms=frag_info["num_heavy_atoms"] if frag_info else None,
                        source_count=frag_info["source_count"] if frag_info else None,
                    )

    # Step 6: Sort by geometric distance, deduplicate by core SMILES, return top-k
    results = sorted(all_results.values(), key=lambda r: r.geometric_distance)
    results = _deduplicate_by_core(results)
    return results[:top_k]


def _strip_brics_labels(smiles: str) -> str:
    """Strip BRICS isotope labels from dummy atoms to get a core SMILES.

    Converts e.g. '[5*]c1ccc([12*])cc1' → '*c1ccc(*)cc1' and re-canonicalizes.
    """
    stripped = re.sub(r'\[\d+\*\]', '[*]', smiles)
    mol = Chem.MolFromSmiles(stripped)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return stripped


def _deduplicate_by_core(results: list[ReplacementResult]) -> list[ReplacementResult]:
    """Remove results that are the same core scaffold differing only in BRICS labels.

    Keeps the best-scoring (lowest geometric_distance) entry per core SMILES.
    Input must already be sorted by geometric_distance.
    """
    seen_cores: dict[str, int] = {}
    deduped = []
    for r in results:
        core = _strip_brics_labels(r.smiles)
        if core not in seen_cores:
            seen_cores[core] = r.fragment_id
            deduped.append(r)
        else:
            logger.debug(
                f"Dedup: fragment {r.fragment_id} ({r.smiles}) is a label variant "
                f"of fragment {seen_cores[core]} (core: {core})"
            )
    if len(results) != len(deduped):
        logger.info(f"Deduplicated {len(results)} → {len(deduped)} results (removed {len(results) - len(deduped)} label variants)")
    return deduped


def _find_cut_bonds(mol: Chem.Mol, match_atoms: set[int]) -> list[CutBond]:
    """Find bonds that cross the boundary between matched and unmatched atoms.

    Only returns heavy-atom cut bonds (skips bonds to hydrogen).
    """
    cut_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in match_atoms) != (a2 in match_atoms):
            matched = a1 if a1 in match_atoms else a2
            external = a2 if a1 in match_atoms else a1
            # Skip H atoms — they're not real attachment points
            if mol.GetAtomWithIdx(external).GetAtomicNum() == 1:
                continue
            if mol.GetAtomWithIdx(matched).GetAtomicNum() == 1:
                continue
            cut_bonds.append(CutBond(
                matched_atom_idx=matched,
                external_atom_idx=external,
                bond_idx=bond.GetIdx(),
                bond_order=int(bond.GetBondTypeAsDouble()),
            ))
    return cut_bonds


def _compute_single_exit_vector(conf, cut_bond: CutBond, match_atoms: set[int]):
    """Compute the exit vector direction for a single cut bond.

    Measured from the substructure side (matching fragment convention):
    origin = matched atom (analogous to fragment neighbor), pointing toward
    the external atom (analogous to fragment dummy).
    """
    pos_ext = np.array(conf.GetAtomPosition(cut_bond.external_atom_idx))
    pos_match = np.array(conf.GetAtomPosition(cut_bond.matched_atom_idx))
    direction = pos_ext - pos_match
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    return ExitVector(origin=pos_match, direction=direction, tip=pos_ext)


def _compute_cut_pair_descriptor(
    conf, cb1: CutBond, cb2: CutBond, match_atoms: set[int]
) -> Optional[VectorPairDescriptor]:
    """Compute geometric descriptor for a pair of cut bonds.

    We measure from the substructure side (matching the DB fragment convention):
    - b1 = matched atom of cut bond 1 (base, analogous to fragment neighbor)
    - t1 = external atom of cut bond 1 (tip, analogous to fragment dummy)
    - b2 = matched atom of cut bond 2
    - t2 = external atom of cut bond 2
    """
    return compute_vector_pair_descriptor(
        conf,
        dummy1_idx=cb1.external_atom_idx,
        neighbor1_idx=cb1.matched_atom_idx,
        dummy2_idx=cb2.external_atom_idx,
        neighbor2_idx=cb2.matched_atom_idx,
    )
