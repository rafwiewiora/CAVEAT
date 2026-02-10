"""Fragment assembly for CAVEAT.

Stitches replacement fragments into parent molecules by removing the
matched substructure and attaching the replacement at the cut points.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers, rdMolTransforms

from caveat.fragment import AttachmentPoint

logger = logging.getLogger(__name__)


def assemble(
    parent_mol: Chem.Mol,
    match_atoms: tuple[int, ...],
    replacement_mol: Chem.Mol,
    replacement_aps: list[AttachmentPoint],
    parent_conf_id: int = 0,
    replacement_conf_id: int = 0,
    optimize: bool = True,
) -> Optional[Chem.Mol]:
    """Assemble a new molecule by replacing matched atoms with a fragment.

    Args:
        parent_mol: the full parent molecule (should have 3D coords)
        match_atoms: tuple of atom indices in parent_mol to remove
        replacement_mol: the replacement fragment (with dummy atoms)
        replacement_aps: attachment points on the replacement fragment
        parent_conf_id: conformer index for the parent
        replacement_conf_id: conformer index for the replacement
        optimize: whether to MMFF-optimize the result

    Returns:
        New molecule with replacement stitched in, or None on failure.
    """
    try:
        return _assemble_impl(
            parent_mol, match_atoms, replacement_mol, replacement_aps,
            parent_conf_id, replacement_conf_id, optimize,
        )
    except Exception as e:
        logger.error(f"Assembly failed: {e}")
        return None


def _assemble_impl(
    parent_mol, match_atoms, replacement_mol, replacement_aps,
    parent_conf_id, replacement_conf_id, optimize,
):
    match_set = set(match_atoms)

    # Find bonds crossing the substructure boundary in the parent
    cut_info = []
    for bond in parent_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            external = a1 if a1 not in match_set else a2
            internal = a2 if a1 not in match_set else a1
            cut_info.append({
                "external_idx": external,
                "internal_idx": internal,
                "bond_type": bond.GetBondType(),
            })

    if len(cut_info) != len(replacement_aps):
        logger.warning(
            f"Mismatch: {len(cut_info)} cut bonds vs {len(replacement_aps)} attachment points"
        )
        # Try to proceed with min of the two
        n_connect = min(len(cut_info), len(replacement_aps))
        cut_info = cut_info[:n_connect]
        replacement_aps = replacement_aps[:n_connect]

    # Build a new editable molecule from the parent, excluding matched atoms
    rw = Chem.RWMol(Chem.Mol())

    # Map: old parent atom idx -> new atom idx
    parent_map = {}
    for atom in parent_mol.GetAtoms():
        if atom.GetIdx() not in match_set:
            new_idx = rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            new_atom = rw.GetAtomWithIdx(new_idx)
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                new_atom.SetChiralTag(atom.GetChiralTag())
            parent_map[atom.GetIdx()] = new_idx

    # Add bonds from the parent (excluding those involving matched atoms)
    for bond in parent_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in parent_map and a2 in parent_map:
            rw.AddBond(parent_map[a1], parent_map[a2], bond.GetBondType())

    # Add the replacement fragment atoms (excluding dummy atoms)
    repl_map = {}
    for atom in replacement_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue  # skip dummy atoms
        new_idx = rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        new_atom = rw.GetAtomWithIdx(new_idx)
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            new_atom.SetChiralTag(atom.GetChiralTag())
        repl_map[atom.GetIdx()] = new_idx

    # Add bonds within the replacement fragment (excluding dummy atom bonds)
    for bond in replacement_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in repl_map and a2 in repl_map:
            rw.AddBond(repl_map[a1], repl_map[a2], bond.GetBondType())

    # Connect the replacement fragment to the parent at the cut points
    for ci, ap in zip(cut_info, replacement_aps):
        parent_ext_new = parent_map[ci["external_idx"]]
        # The attachment point's neighbor is the real atom in the fragment
        if ap.neighbor_atom_idx in repl_map:
            repl_neighbor_new = repl_map[ap.neighbor_atom_idx]
            rw.AddBond(parent_ext_new, repl_neighbor_new, ci["bond_type"])
        else:
            logger.warning(f"Attachment point neighbor {ap.neighbor_atom_idx} not in replacement map")

    # Sanitize
    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        logger.warning(f"Sanitization failed: {e}")
        return None

    result_mol = rw.GetMol()

    # Generate 3D coordinates
    result_mol = _embed_assembled(result_mol, parent_mol, parent_map,
                                  parent_conf_id, optimize)

    return result_mol


def _embed_assembled(
    result_mol: Chem.Mol,
    parent_mol: Chem.Mol,
    parent_map: dict[int, int],
    parent_conf_id: int,
    optimize: bool,
) -> Chem.Mol:
    """Generate 3D coordinates for the assembled molecule.

    Uses constrained embedding: fix parent atom positions and embed new atoms.
    """
    result_mol = Chem.AddHs(result_mol)

    # Try constrained embedding using parent coordinates as reference
    if parent_mol.GetNumConformers() > 0:
        parent_conf = parent_mol.GetConformer(parent_conf_id)

        # Create coordinate map: new atom idx -> Point3D from parent
        from rdkit.Geometry import Point3D
        coord_map = {}
        for old_idx, new_idx in parent_map.items():
            if new_idx < result_mol.GetNumAtoms():
                pos = parent_conf.GetAtomPosition(old_idx)
                coord_map[new_idx] = Point3D(pos.x, pos.y, pos.z)

        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = False
        params.SetCoordMap(coord_map)

        cid = AllChem.EmbedMolecule(result_mol, params)
        if cid < 0:
            # Fallback without constraints
            params2 = rdDistGeom.ETKDGv3()
            params2.randomSeed = 42
            params2.useRandomCoords = True
            cid = AllChem.EmbedMolecule(result_mol, params2)
    else:
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(result_mol, params)

    if cid < 0:
        logger.warning("Could not embed assembled molecule")
        return Chem.RemoveHs(result_mol)

    if optimize:
        try:
            rdForceFieldHelpers.MMFFOptimizeMolecule(result_mol, confId=cid)
        except Exception:
            try:
                rdForceFieldHelpers.UFFOptimizeMolecule(result_mol, confId=cid)
            except Exception:
                pass

    result_mol = Chem.RemoveHs(result_mol)
    return result_mol
