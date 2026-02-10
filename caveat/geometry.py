"""Bond vector geometry computation for CAVEAT.

Computes 3D geometric descriptors of fragment attachment points following
the CAVEAT parameterization (Lauri & Bartlett, 1994):
  - d: distance between base atoms b1, b2
  - alpha1: angle t1-b1-b2
  - alpha2: angle b1-b2-t2
  - delta: dihedral t1-b1-b2-t2

Where b1,b2 are the real atoms at attachment points and t1,t2 are the
tip (dummy) atom positions representing the exit vector directions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D


@dataclass
class ExitVector:
    """A single exit vector at an attachment point."""
    origin: np.ndarray  # position of the base (real neighbor) atom
    direction: np.ndarray  # unit direction vector pointing outward
    tip: np.ndarray  # position of the tip (dummy atom or extrapolated point)


@dataclass
class VectorPairDescriptor:
    """CAVEAT-style geometric descriptor for a pair of attachment points."""
    distance: float  # d(b1, b2)
    angle1: float  # angle t1-b1-b2 in degrees
    angle2: float  # angle b1-b2-t2 in degrees
    dihedral: float  # dihedral t1-b1-b2-t2 in degrees [-180, 180]

    def as_array(self) -> np.ndarray:
        return np.array([self.distance, self.angle1, self.angle2, self.dihedral])


def embed_fragment(mol: Chem.Mol, n_confs: int = 10, random_seed: int = 42) -> Chem.Mol:
    """Generate 3D conformers for a fragment.

    Adds hydrogens, generates conformers with ETKDGv3, and MMFF-optimizes them.
    Returns a new mol with Hs and 3D conformers. Dummy atoms (*) are kept.
    """
    mol = Chem.RWMol(mol)
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    params.pruneRmsThresh = 0.5

    n_generated = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if n_generated == 0:
        # Fallback: try with less strict parameters
        params.useRandomCoords = True
        n_generated = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if n_generated == 0:
        raise ValueError(f"Could not embed fragment: {Chem.MolToSmiles(mol)}")

    # MMFF optimize each conformer
    try:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    except Exception:
        # If MMFF fails, try UFF
        try:
            results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            results = []  # keep unoptimized conformers

    return mol


def get_exit_vector(conf: Chem.Conformer, dummy_idx: int, neighbor_idx: int) -> ExitVector:
    """Compute the exit vector at an attachment point.

    The exit vector points from the neighbor (real) atom outward through
    the dummy atom position.

    Args:
        conf: RDKit conformer with 3D coordinates
        dummy_idx: atom index of the dummy atom (*)
        neighbor_idx: atom index of the real atom bonded to the dummy
    """
    pos_dummy = np.array(conf.GetAtomPosition(dummy_idx))
    pos_neighbor = np.array(conf.GetAtomPosition(neighbor_idx))

    direction = pos_dummy - pos_neighbor
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("Dummy and neighbor atoms have identical positions")
    direction = direction / norm

    return ExitVector(
        origin=pos_neighbor,
        direction=direction,
        tip=pos_dummy,
    )


def compute_vector_pair_descriptor(
    conf: Chem.Conformer,
    dummy1_idx: int,
    neighbor1_idx: int,
    dummy2_idx: int,
    neighbor2_idx: int,
) -> VectorPairDescriptor:
    """Compute CAVEAT geometric descriptor for a pair of attachment points.

    Args:
        conf: conformer with 3D coordinates
        dummy1_idx, neighbor1_idx: first attachment point (tip, base)
        dummy2_idx, neighbor2_idx: second attachment point (tip, base)

    Returns:
        VectorPairDescriptor with distance, angle1, angle2, dihedral
    """
    t1 = np.array(conf.GetAtomPosition(dummy1_idx))
    b1 = np.array(conf.GetAtomPosition(neighbor1_idx))
    b2 = np.array(conf.GetAtomPosition(neighbor2_idx))
    t2 = np.array(conf.GetAtomPosition(dummy2_idx))

    # Distance between base atoms
    d = float(np.linalg.norm(b2 - b1))

    # Angle t1-b1-b2
    angle1 = _angle(t1, b1, b2)

    # Angle b1-b2-t2
    angle2 = _angle(b1, b2, t2)

    # Dihedral t1-b1-b2-t2
    dihedral = _dihedral(t1, b1, b2, t2)

    return VectorPairDescriptor(
        distance=d,
        angle1=angle1,
        angle2=angle2,
        dihedral=dihedral,
    )


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle a-b-c in degrees."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute dihedral angle a-b-c-d in degrees [-180, 180]."""
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return float(np.degrees(np.arctan2(-y, x)))


def canonicalize_dihedral(dihedral: float) -> float:
    """Canonicalize dihedral to handle enantiomeric sign convention.

    We use the convention that the dihedral is in [0, 180] by taking
    the absolute value. This means enantiomeric fragments will match.
    For applications needing chirality, use the raw dihedral.
    """
    return abs(dihedral)
