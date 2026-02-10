"""Fragmentation engine for CAVEAT.

Implements BRICS-based fragmentation of molecules into fragments with
tracked attachment points. Extensible via the Fragmenter base class.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import BRICS


@dataclass
class AttachmentPoint:
    """An attachment point on a fragment where it was cut from the parent."""
    dummy_atom_idx: int  # index of the dummy atom (*) in the fragment
    neighbor_atom_idx: int  # index of the real atom bonded to the dummy
    brics_label: int  # BRICS environment label (isotope number on dummy)
    bond_order: int = 1  # bond order of the original bond (typically 1 for BRICS)

    def to_dict(self) -> dict:
        return {
            "dummy_atom_idx": self.dummy_atom_idx,
            "neighbor_atom_idx": self.neighbor_atom_idx,
            "brics_label": self.brics_label,
            "bond_order": self.bond_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AttachmentPoint:
        return cls(**d)


@dataclass
class Fragment:
    """A molecular fragment with attachment point information."""
    mol: Chem.Mol
    smiles: str  # canonical SMILES (with dummy atoms)
    attachment_points: list[AttachmentPoint]
    source_smiles: str = ""  # SMILES of the source molecule
    brics_labels: list[int] = field(default_factory=list)

    @property
    def num_attachment_points(self) -> int:
        return len(self.attachment_points)

    @property
    def num_heavy_atoms(self) -> int:
        """Count heavy atoms excluding dummy atoms."""
        return sum(
            1 for atom in self.mol.GetAtoms()
            if atom.GetAtomicNum() > 0  # exclude dummy atoms (atomic num 0)
        )


class Fragmenter(ABC):
    """Abstract base class for fragmentation methods."""

    @abstractmethod
    def fragment(self, mol: Chem.Mol, source_smiles: str = "") -> list[Fragment]:
        """Fragment a molecule into a list of Fragments."""
        ...


class BRICSFragmenter(Fragmenter):
    """BRICS-based fragmentation.

    Uses RDKit's BRICS implementation to identify retrosynthetically
    sensible bond cuts and produce fragments with labeled attachment points.
    """

    def __init__(self, min_heavy_atoms: int = 3):
        self.min_heavy_atoms = min_heavy_atoms

    def fragment(self, mol: Chem.Mol, source_smiles: str = "") -> list[Fragment]:
        if mol is None:
            return []

        if not source_smiles:
            source_smiles = Chem.MolToSmiles(mol)

        # Find BRICS bonds
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:
            return []

        # Extract bond indices and labels
        bond_indices = []
        bond_labels = []
        for (i, j), (label_i, label_j) in brics_bonds:
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bond_indices.append(bond.GetIdx())
            # Labels are strings like "1", "3", etc.
            bond_labels.append((int(label_i), int(label_j)))

        if not bond_indices:
            return []

        # Fragment the molecule on the identified bonds
        # dummyLabels assigns isotope numbers to dummy atoms for tracking
        dummy_labels = []
        for li, lj in bond_labels:
            dummy_labels.append((li, lj))

        frag_mol = Chem.FragmentOnBonds(
            mol,
            bond_indices,
            dummyLabels=dummy_labels,
        )

        # Split into individual fragment molecules
        frag_atom_mapping = []
        frag_mols = Chem.GetMolFrags(
            frag_mol,
            asMols=True,
            sanitizeFrags=True,
            fragsMolAtomMapping=frag_atom_mapping,
        )

        fragments = []
        for fmol in frag_mols:
            # Identify attachment points (dummy atoms)
            aps = []
            for atom in fmol.GetAtoms():
                if atom.GetAtomicNum() == 0:  # dummy atom
                    isotope = atom.GetIsotope()
                    # Find the neighbor (real atom bonded to dummy)
                    neighbors = atom.GetNeighbors()
                    if neighbors:
                        neighbor = neighbors[0]
                        aps.append(AttachmentPoint(
                            dummy_atom_idx=atom.GetIdx(),
                            neighbor_atom_idx=neighbor.GetIdx(),
                            brics_label=isotope,
                            bond_order=1,
                        ))

            # Count heavy atoms (non-dummy)
            n_heavy = sum(1 for a in fmol.GetAtoms() if a.GetAtomicNum() > 0)
            if n_heavy < self.min_heavy_atoms:
                continue

            smiles = Chem.MolToSmiles(fmol)
            brics_labels_list = sorted(set(ap.brics_label for ap in aps))

            fragments.append(Fragment(
                mol=fmol,
                smiles=smiles,
                attachment_points=aps,
                source_smiles=source_smiles,
                brics_labels=brics_labels_list,
            ))

        return fragments
