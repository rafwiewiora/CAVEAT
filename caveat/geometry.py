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


def align_single_vector(
    frag_positions: np.ndarray,
    frag_neighbor_pos: np.ndarray,
    frag_dummy_pos: np.ndarray,
    target_pos: np.ndarray,
    target_direction_pos: np.ndarray,
) -> np.ndarray:
    """Compute rigid-body transform to align a 1-AP fragment to a target position.

    Translates the fragment so its AP neighbor lands at target_pos, then rotates
    so the exit vector (neighbor→dummy) aligns with the target direction
    (external→internal on parent). Uses Rodrigues' rotation formula.

    Args:
        frag_positions: Nx3 array of all fragment atom positions
        frag_neighbor_pos: position of the fragment's AP neighbor atom
        frag_dummy_pos: position of the fragment's dummy atom
        target_pos: where the AP neighbor should end up (parent external atom)
        target_direction_pos: direction target (parent internal/matched atom)

    Returns:
        4x4 homogeneous transformation matrix
    """
    # Fragment exit vector: neighbor → dummy
    frag_vec = frag_dummy_pos - frag_neighbor_pos
    frag_vec_norm = np.linalg.norm(frag_vec)
    if frag_vec_norm < 1e-8:
        return np.eye(4)
    frag_vec = frag_vec / frag_vec_norm

    # Target direction: external → internal (matching the exit vector convention)
    target_vec = target_direction_pos - target_pos
    target_vec_norm = np.linalg.norm(target_vec)
    if target_vec_norm < 1e-8:
        return np.eye(4)
    target_vec = target_vec / target_vec_norm

    # Step 1: Translation to move fragment neighbor to origin
    T1 = np.eye(4)
    T1[:3, 3] = -frag_neighbor_pos

    # Step 2: Rotation to align exit vectors (Rodrigues' formula)
    R = _rotation_matrix_between_vectors(frag_vec, target_vec)

    # Step 3: Translation to target position
    T2 = np.eye(4)
    T2[:3, 3] = target_pos

    # Combined: T2 @ R @ T1
    transform = T2 @ R @ T1
    return transform


def align_two_vectors(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Compute rigid-body transform using Kabsch (SVD) alignment.

    For 2-AP fragments, aligns 4 point pairs:
    (neighbor1, dummy1, neighbor2, dummy2) on source and target.

    Args:
        source_points: Nx3 array of source points (N >= 3)
        target_points: Nx3 array of target points (same N)

    Returns:
        4x4 homogeneous transformation matrix
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[0] >= 2

    # Centroids
    src_centroid = source_points.mean(axis=0)
    tgt_centroid = target_points.mean(axis=0)

    # Center the points
    src_centered = source_points - src_centroid
    tgt_centered = target_points - tgt_centroid

    # Kabsch: compute optimal rotation via SVD
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Build 4x4 transform
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = tgt_centroid - R @ src_centroid
    return transform


def apply_transform(positions: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid-body transform to an Nx3 array of positions.

    Args:
        positions: Nx3 array of 3D positions
        transform: 4x4 homogeneous transformation matrix

    Returns:
        Nx3 array of transformed positions
    """
    N = positions.shape[0]
    # Convert to homogeneous coordinates
    ones = np.ones((N, 1))
    homogeneous = np.hstack([positions, ones])  # Nx4
    transformed = (transform @ homogeneous.T).T  # Nx4
    return transformed[:, :3]


def _rotation_matrix_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute 4x4 rotation matrix that rotates v1 onto v2 using Rodrigues' formula.

    Args:
        v1, v2: unit vectors (3D)

    Returns:
        4x4 homogeneous rotation matrix
    """
    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2) + 1e-10)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    sin_angle = np.linalg.norm(cross)

    if sin_angle < 1e-8:
        # Vectors are parallel
        if dot > 0:
            return np.eye(4)  # same direction
        else:
            # Opposite direction — rotate 180° around any perpendicular axis
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v1, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(v1, perp)
            axis = axis / np.linalg.norm(axis)
            # 180° rotation: R = 2 * outer(axis, axis) - I
            R3 = 2.0 * np.outer(axis, axis) - np.eye(3)
            R = np.eye(4)
            R[:3, :3] = R3
            return R

    # Rodrigues' formula: R = I + K + K²/(1+c), where K is skew-symmetric of cross
    axis = cross / sin_angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R3 = np.eye(3) + K + (K @ K) * (1.0 / (1.0 + dot))
    R = np.eye(4)
    R[:3, :3] = R3
    return R


def canonicalize_dihedral(dihedral: float) -> float:
    """Canonicalize dihedral to handle enantiomeric sign convention.

    We use the convention that the dihedral is in [0, 180] by taking
    the absolute value. This means enantiomeric fragments will match.
    For applications needing chirality, use the raw dihedral.
    """
    return abs(dihedral)
