"""Query engine for CAVEAT.

Given a molecule and a substructure to replace, find compatible replacement
fragments from the database ranked by geometric similarity.
"""

from __future__ import annotations

import logging
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
    use_kdtree: bool = True,
    mol_smiles: Optional[str] = None,
) -> list[ReplacementResult]:
    """Find replacement fragments for a substructure in a molecule.

    Args:
        mol: the query molecule
        query_smarts: SMARTS pattern identifying the substructure to replace
        db: the fragment database
        n_confs: number of conformers for the query molecule
        top_k: maximum number of results to return
        tolerance: geometric tolerance multiplier (1.0 = default tolerances)
        use_kdtree: use KDTree for fast search (True) or SQL range query (False)
        mol_smiles: optional SMILES for the query molecule

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

    # Step 3: Embed the query molecule in 3D
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
            # For single AP fragments, query all 1-AP fragments
            # No geometric pair to match — rank by other criteria
            rows = db.conn.execute(
                "SELECT id, canonical_smiles, num_heavy_atoms, source_count FROM fragments WHERE num_attachment_points = 1"
            ).fetchall()
            for row in rows:
                fid, smi, nha, sc = row
                if fid not in all_results:
                    all_results[fid] = ReplacementResult(
                        fragment_id=fid,
                        smiles=smi,
                        geometric_distance=0.0,
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

    # Step 6: Sort by geometric distance and return top-k
    results = sorted(all_results.values(), key=lambda r: r.geometric_distance)
    return results[:top_k]


def _find_cut_bonds(mol: Chem.Mol, match_atoms: set[int]) -> list[CutBond]:
    """Find bonds that cross the boundary between matched and unmatched atoms."""
    cut_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in match_atoms) != (a2 in match_atoms):
            matched = a1 if a1 in match_atoms else a2
            external = a2 if a1 in match_atoms else a1
            cut_bonds.append(CutBond(
                matched_atom_idx=matched,
                external_atom_idx=external,
                bond_idx=bond.GetIdx(),
                bond_order=int(bond.GetBondTypeAsDouble()),
            ))
    return cut_bonds


def _compute_single_exit_vector(conf, cut_bond: CutBond, match_atoms: set[int]):
    """Compute the exit vector direction for a single cut bond."""
    pos_ext = np.array(conf.GetAtomPosition(cut_bond.external_atom_idx))
    pos_match = np.array(conf.GetAtomPosition(cut_bond.matched_atom_idx))
    direction = pos_match - pos_ext
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    return ExitVector(origin=pos_ext, direction=direction, tip=pos_match)


def _compute_cut_pair_descriptor(
    conf, cb1: CutBond, cb2: CutBond, match_atoms: set[int]
) -> Optional[VectorPairDescriptor]:
    """Compute geometric descriptor for a pair of cut bonds.

    We treat:
    - b1 = external atom of cut bond 1 (base atom of exit vector 1)
    - t1 = matched atom of cut bond 1 (tip/dummy direction)
    - b2 = external atom of cut bond 2
    - t2 = matched atom of cut bond 2
    """
    return compute_vector_pair_descriptor(
        conf,
        dummy1_idx=cb1.matched_atom_idx,
        neighbor1_idx=cb1.external_atom_idx,
        dummy2_idx=cb2.matched_atom_idx,
        neighbor2_idx=cb2.external_atom_idx,
    )
