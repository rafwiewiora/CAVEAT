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
from rdkit.Geometry import Point3D

from caveat.fragment import AttachmentPoint
from caveat.geometry import align_single_vector, align_two_vectors, apply_transform

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

    # Expand match_set to include H atoms exclusively bonded to matched atoms.
    # When parent has explicit Hs, the SMARTS match only captures heavy atoms,
    # leaving orphan Hs that would become disconnected fragments.
    for atom in parent_mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIdx() not in match_set:
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1 and neighbors[0].GetIdx() in match_set:
                match_set.add(atom.GetIdx())

    # Find bonds crossing the substructure boundary in the parent
    # Only count heavy-atom cut bonds (skip H-to-external bonds)
    cut_info = []
    for bond in parent_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if (a1 in match_set) != (a2 in match_set):
            external = a1 if a1 not in match_set else a2
            internal = a2 if a1 not in match_set else a1
            # Skip if the external atom is hydrogen — not a real attachment
            if parent_mol.GetAtomWithIdx(external).GetAtomicNum() == 1:
                continue
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

    # Generate 3D coordinates — prefer direct placement over re-embedding
    parent_has_3d = parent_mol.GetNumConformers() > 0
    repl_has_3d = replacement_mol.GetNumConformers() > 0

    if parent_has_3d and repl_has_3d:
        result_mol = _place_coordinates(
            result_mol, parent_mol, replacement_mol,
            parent_map, repl_map, cut_info, replacement_aps,
            parent_conf_id, replacement_conf_id, optimize,
        )
    else:
        # Fallback: re-embed from scratch
        result_mol = _embed_assembled(result_mol, parent_mol, parent_map,
                                      parent_conf_id, optimize)

    return result_mol


def _place_coordinates(
    result_mol: Chem.Mol,
    parent_mol: Chem.Mol,
    replacement_mol: Chem.Mol,
    parent_map: dict[int, int],
    repl_map: dict[int, int],
    cut_info: list[dict],
    replacement_aps: list[AttachmentPoint],
    parent_conf_id: int,
    replacement_conf_id: int,
    optimize: bool,
) -> Chem.Mol:
    """Place coordinates by copying parent atoms and aligning fragment atoms.

    Parent atom positions are copied directly. Fragment atom positions are
    rigidly transformed from the stored DB conformer to align with the
    parent's cut-point vectors.
    """
    parent_conf = parent_mol.GetConformer(parent_conf_id)
    repl_conf = replacement_mol.GetConformer(replacement_conf_id)
    n_atoms = result_mol.GetNumAtoms()

    # Create a new conformer for the result
    conf = Chem.Conformer(n_atoms)

    # Step 1: Copy parent atom positions
    for old_idx, new_idx in parent_map.items():
        if new_idx < n_atoms:
            pos = parent_conf.GetAtomPosition(old_idx)
            conf.SetAtomPosition(new_idx, Point3D(pos.x, pos.y, pos.z))

    # Step 2: Gather fragment atom positions from the replacement conformer
    repl_atom_indices = sorted(repl_map.keys())  # original indices in replacement mol
    repl_positions = np.array([
        list(repl_conf.GetAtomPosition(idx)) for idx in repl_atom_indices
    ])
    repl_idx_to_row = {idx: row for row, idx in enumerate(repl_atom_indices)}

    # Step 3: Compute alignment transform
    n_aps = len(cut_info)
    if n_aps == 1:
        ci = cut_info[0]
        ap = replacement_aps[0]

        # Fragment side: neighbor and dummy positions from the replacement conformer
        frag_neighbor_pos = np.array(list(repl_conf.GetAtomPosition(ap.neighbor_atom_idx)))
        frag_dummy_pos = np.array(list(repl_conf.GetAtomPosition(ap.dummy_atom_idx)))

        # Target side: external atom pos (where neighbor should go) and
        # internal atom pos (direction the exit vector should point)
        target_pos = np.array(list(parent_conf.GetAtomPosition(ci["external_idx"])))
        target_dir_pos = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))

        transform = align_single_vector(
            repl_positions, frag_neighbor_pos, frag_dummy_pos,
            target_pos, target_dir_pos,
        )

    elif n_aps >= 2:
        # Build point pairs for Kabsch alignment
        # For each cut bond / AP pair: neighbor→external, dummy→internal
        source_pts = []
        target_pts = []
        for ci, ap in zip(cut_info, replacement_aps):
            frag_neighbor_pos = np.array(list(repl_conf.GetAtomPosition(ap.neighbor_atom_idx)))
            frag_dummy_pos = np.array(list(repl_conf.GetAtomPosition(ap.dummy_atom_idx)))
            target_ext = np.array(list(parent_conf.GetAtomPosition(ci["external_idx"])))
            target_int = np.array(list(parent_conf.GetAtomPosition(ci["internal_idx"])))

            source_pts.append(frag_neighbor_pos)
            source_pts.append(frag_dummy_pos)
            target_pts.append(target_ext)
            target_pts.append(target_int)

        transform = align_two_vectors(
            np.array(source_pts),
            np.array(target_pts),
        )
    else:
        transform = np.eye(4)

    # Step 4: Apply transform to fragment positions and set on conformer
    transformed = apply_transform(repl_positions, transform)
    for orig_idx, row_idx in repl_idx_to_row.items():
        new_idx = repl_map[orig_idx]
        if new_idx < n_atoms:
            pos = transformed[row_idx]
            conf.SetAtomPosition(new_idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))

    conf.SetId(0)
    result_mol.AddConformer(conf, assignId=True)

    # Step 5: Optional junction-only MMFF minimization
    if optimize:
        try:
            _optimize_junction(result_mol, parent_map, repl_map, n_atoms)
        except Exception:
            pass  # keep unoptimized coordinates

    return result_mol


def _optimize_junction(
    mol: Chem.Mol,
    parent_map: dict[int, int],
    repl_map: dict[int, int],
    n_atoms: int,
):
    """MMFF minimize only atoms near the junction (within 1 bond of cut point).

    All parent atoms and most fragment atoms are fixed; only atoms directly
    at or adjacent to the junction bonds are allowed to move.
    """
    # Find junction atoms in the result mol: atoms that were connected across
    # parent ↔ fragment boundary
    junction_atoms = set()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        parent_new = set(parent_map.values())
        repl_new = set(repl_map.values())
        # If this atom is in one set and has a neighbor in the other, it's a junction atom
        if idx in parent_new and any(n in repl_new for n in neighbors):
            junction_atoms.add(idx)
            junction_atoms.update(n for n in neighbors if n in repl_new)
        elif idx in repl_new and any(n in parent_new for n in neighbors):
            junction_atoms.add(idx)
            junction_atoms.update(n for n in neighbors if n in parent_new)

    if not junction_atoms:
        return

    # Add Hs for MMFF, optimize with fixed atoms, then remove Hs
    mol_h = Chem.AddHs(mol, addCoords=True)
    try:
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_h)
        if mp is None:
            return
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_h, mp, confId=0)
        if ff is None:
            return

        # Fix all heavy atoms except junction atoms
        for i in range(n_atoms):
            if i not in junction_atoms:
                ff.AddFixedPoint(i)

        ff.Minimize(maxIts=200)

        # Copy back the optimized positions for junction atoms
        conf_h = mol_h.GetConformer(0)
        conf = mol.GetConformer(0)
        for idx in junction_atoms:
            if idx < n_atoms:
                pos = conf_h.GetAtomPosition(idx)
                conf.SetAtomPosition(idx, Point3D(pos.x, pos.y, pos.z))
    except Exception:
        pass


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
