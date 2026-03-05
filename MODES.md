# CAVEAT Screening Modes & Fragmentation Methods

## Overview

CAVEAT replaces a substructure in a parent molecule with geometrically compatible fragments from a database. The pipeline has two independent axes of variation:

1. **Replacement mode** — how many bonds are cut between the substructure and the rest of the molecule (1-AP vs 2-AP)
2. **Fragmentation method** — how the database fragments were generated (BRICS, reactive handles, C-H enumeration)

These are orthogonal: any fragmentation method can produce fragments for either replacement mode.

```
                        ┌─────────────────────────────────────────────┐
                        │           CAVEAT Pipeline                   │
                        │                                             │
                        │   Parent Molecule + Substructure to Replace │
                        │                   │                         │
                        │          ┌────────┴────────┐                │
                        │          ▼                  ▼                │
                        │     ┌─────────┐      ┌──────────┐          │
                        │     │  1-AP   │      │   2-AP   │          │
                        │     │ (1 cut) │      │ (2 cuts) │          │
                        │     └────┬────┘      └────┬─────┘          │
                        │          │                 │                │
                        │          ▼                 ▼                │
                        │   ┌─────────────────────────────┐          │
                        │   │      Fragment Database       │          │
                        │   │  BRICS │ Reactive │ C-H enum │          │
                        │   └──────────────┬──────────────┘          │
                        │                  ▼                          │
                        │        Geometric Matching +                 │
                        │        3D Assembly                          │
                        │                  │                          │
                        │                  ▼                          │
                        │         Assembled Products                  │
                        └─────────────────────────────────────────────┘
```

---

## Replacement Modes

### 2-AP (Two Attachment Points)

Cut **two bonds** between the substructure and the parent, replacing an internal linker or scaffold element.

```
    2-AP Replacement: Internal linker swap

    ╔═══════╗       ╔═══════╗               ╔═══════╗       ╔═══════╗
    ║       ║       ║       ║               ║       ║       ║       ║
    ║ Core  ║──[*]══[linker]══[*]──║  Side  ║   ══► ║ Core  ║──[*]══[ new  ]══[*]──║  Side  ║
    ║       ║       ║       ║               ║       ║       ║       ║
    ╚═══════╝       ╚═══════╝               ╚═══════╝       ╚═══════╝
                    ▲                                        ▲
                    │ remove                                  │ insert
                    │                                        │
                    └── substructure match                    └── database fragment

    Geometric matching:     4D descriptor between attachment vectors
                            (distance, angle1, angle2, dihedral)

                          [*]─── A ···distance··· B ───[*]
                               ╱                     ╲
                          angle1                       angle2
                             ╱                           ╲
                        ext_A                             ext_B
```

- The query substructure must have exactly 2 bonds crossing the boundary
- Fragments are matched using a **4D geometric descriptor**: (distance, angle1, angle2, dihedral) between the two attachment vector pairs
- Indexed via KD-tree in normalized 4D space for fast nearest-neighbor lookup
- Typical use case: replacing a heterocyclic linker between two larger ring systems

**CLI example** (replacing the piperazine linker in imatinib):
```bash
caveat query --db fragments.db \
  --mol imatinib.sdf \
  --replace "C1CNCCN1" \
  --top 50
```

### 1-AP (One Attachment Point)

Cut **one bond**, replacing an entire pendant group (sidechain, tail, cap).

```
    1-AP Replacement: Sidechain swap

    ╔═══════╗                               ╔═══════╗
    ║       ║                               ║       ║
    ║ Core  ║──[*]══[sidechain]         ══► ║ Core  ║──[*]══[ new fragment ]
    ║       ║                               ║       ║
    ╚═══════╝                               ╚═══════╝
                    ▲                                        ▲
                    │ remove                                  │ insert
                    │                                        │
                    └── substructure match                    └── database fragment
```

- The query substructure has exactly 1 bond crossing the boundary
- No geometric pair to compare — fragments are ranked by **BRICS label compatibility** and **size similarity**
- Optionally constrained by a **pharmacophore filter**:

```
    Pharmacophore-constrained 1-AP:

    Original sidechain:                  Replacement must match:

       ┌─ HBA (e.g. pyridine N)            ┌─ HBA within tolerance
       │                                    │
       ●  ← 3D reference position          ●  ← must land near here
      ╱                                    ╱
     ╱  dist ± tol                        ╱
    ╱   angle ± tol                      ╱
   [*]──── Core                         [*]──── Core
```

- Post-assembly **3D pharmacophore verification**: after stitching the fragment in, check that the pharmacophore feature lands near the reference position in 3D space

**CLI example** (replacing the N-methylpiperazine sidechain in imatinib):
```bash
caveat query --db fragments.db \
  --mol imatinib.sdf \
  --replace "CN1CCNCC1" \
  --top 50
```

### Combination Mode (2-AP x 1-AP / 2-AP x 2-AP)

Combine two fragments to build novel replacements not found in any single database:

```
    Combination: 2-AP linker + 1-AP cap → novel sidechain

    Database A (2-AP linkers):        Database B (1-AP caps):

       [*]──[ frag A ]──[*]             [*]──[ frag B ]

                         ╲                ╱
                          ╲──── join ────╱
                                │
                                ▼

    Result:    Core ──[*]──[ frag A ]──[ frag B ]

    ─────────────────────────────────────────────────

    Combination: 2-AP + 2-AP → extended linker

       [*]──[ frag A ]──[*]    +    [*]──[ frag B ]──[*]

                                │
                                ▼

    Result:    Core ──[*]──[ frag A ]──[ frag B ]──[*]── Side
```

- **2-AP x 1-AP**: Use a 2-AP fragment as a linker and cap it with a 1-AP fragment to create a novel sidechain replacement
- **2-AP x 2-AP**: Concatenate two 2-AP fragments for longer/more complex linkers
- This dramatically expands chemical space beyond what's in any single database

---

## Geometric Matching (KD-Tree Search)

For 2-AP fragments, CAVEAT computes a **4D geometric descriptor** from the two attachment vectors and uses a KD-tree for fast nearest-neighbor lookup across millions of fragments.

### The 4D Descriptor

Each pair of attachment vectors defines four measurements:

```
    Attachment vector pair on a fragment:

                        exit vector 1               exit vector 2
                             ↑                           ↑
                           ext_A                       ext_B
                            ╱                             ╲
                     angle1╱                               ╲angle2
                          ╱                                 ╲
               [*]───── A ─ ─ ─ ─ distance ─ ─ ─ ─ B ─────[*]
                         ╲                           ╱
                          ╲       dihedral          ╱
                           ╲    (twist angle)      ╱
                            ╲       ╱             ╱
                             plane1    plane2

    4D descriptor = (distance, angle1, angle2, dihedral)

        distance:  Å between neighbor atoms A and B
        angle1:    degrees, ext_A – A – B
        angle2:    degrees, A – B – ext_B
        dihedral:  degrees, ext_A – A – B – ext_B (out-of-plane twist)
```

### KD-Tree Indexing

All fragment descriptors are stored in a **KD-tree** (k-dimensional tree), a spatial data structure that enables O(log N) nearest-neighbor queries instead of O(N) brute-force search.

```
    Building the index:

    For each fragment × each conformer:
        compute 4D descriptor (dist, a1, a2, dih)
              │
              ▼
        normalize:  [dist/1, a1/15, a2/15, dih/30]     ← weight angles less
              │                                            (15° ~ 1 Å, 30° ~ 1 Å)
              ▼
        insert into KD-tree

    ─────────────────────────────────────────────────────

    Querying:

    Substructure to replace
              │
              ▼
    Compute reference 4D descriptor from parent 3D coords
              │
              ▼
    Normalize with same weights
              │
              ▼
    KD-tree nearest-neighbor search (k neighbors)
              │
              ▼
    Rank by Euclidean distance in normalized space
              │
              ▼
    Return top-k fragments sorted by geometric similarity score
```

The normalization weights (`distance/1`, `angle/15`, `dihedral/30`) reflect that a 15-degree angle deviation is roughly equivalent to a 1 Angstrom distance deviation for typical drug-like fragments, and dihedrals are more tolerant.

---

## 3D Fragment Alignment & Assembly

After identifying geometrically compatible fragments, CAVEAT must **align** the fragment's 3D coordinates to the parent molecule and **stitch** them together. Different algorithms are used for 1-AP vs 2-AP:

### 1-AP Alignment: Rodrigues' Rotation

For single-attachment-point fragments, there's only one bond vector to match. The alignment uses **Rodrigues' rotation formula** — a closed-form method to compute the rotation matrix that maps one unit vector onto another.

```
    1-AP alignment steps:

    Step 1: Translate                    Step 2: Rotate (Rodrigues)

    Fragment:    [*]──N──(rest)          Fragment exit vector:  N → [*]
                                         Target direction:      internal → external
         move N to                            ↓
         parent internal atom pos      Rodrigues' formula finds rotation R
                 │                     that maps one vector onto the other:
                 ▼
    N lands at ● (internal atom)            axis = frag_vec × target_vec
    [*] points wrong direction              angle = arccos(frag_vec · target_vec)
                                            R = I + sin(θ)K + (1-cos(θ))K²
                 │                              where K = skew-symmetric(axis)
                 ▼
                                    Step 3: Refine torsion around bond axis

                                    Single-vector alignment leaves one degree
                                    of freedom: rotation around the bond axis.

                                    Scan torsion angles, pick the one that
                                    best matches the parent's local geometry.

         ● ─── N ──(rest)               ● ─── N ──(rest)
        ╱       ↻                      ╱
       ext    rotate around           ext     final
              ●─N axis                        orientation
```

### 2+ AP Alignment: Kabsch Algorithm (SVD)

For two or more attachment points, we have enough point correspondences for a **least-squares optimal rotation**. The Kabsch algorithm uses Singular Value Decomposition (SVD) to find the rotation matrix that minimizes RMSD between paired points.

```
    2-AP alignment via Kabsch:

    Source points (fragment):           Target points (parent):

       ext_A ── neighbor_A                 external_1 ── internal_1
                     │                                       │
               [fragment body]                        [cut bonds in parent]
                     │                                       │
       ext_B ── neighbor_B                 external_2 ── internal_2

    Point pairs to align:
       neighbor_A  →  internal_1     (fragment AP neighbor → parent internal)
       dummy_A     →  external_1     (fragment dummy → parent external)
       neighbor_B  →  internal_2
       dummy_B     →  external_2

    ──────────────────────────────────────────────────

    Kabsch algorithm:

    1. Compute centroids of source and target point sets
    2. Center both sets (subtract centroids)
    3. Compute cross-covariance matrix:   H = source_centered^T × target_centered
    4. SVD decomposition:                 H = U × S × V^T
    5. Optimal rotation:                  R = V × diag(1,1,det(VU^T)) × U^T
    6. Translation:                       t = target_centroid - R × source_centroid

    The diag(1,1,det) term prevents reflections (ensures proper rotation).
    Result: rigid-body transform (R, t) that optimally superimposes
    the fragment attachment vectors onto the parent cut bonds.
```

### Assembly (Stitching)

After alignment, the fragment is stitched into the parent:

```
    Assembly process:

    1. Copy all parent atoms (except matched substructure) with original coordinates
    2. Copy all fragment atoms (except dummies) with aligned coordinates
    3. Create new bonds at each junction:
       parent external atom ──── fragment neighbor atom
    4. Optional: MMFF force-field minimization of junction region
       (atoms within 2 bonds of cut point can relax, everything else fixed)
    5. Enforce coplanarity at aromatic ring junctions
       (project fragment atoms onto ring plane if needed)
```

---

## Fragmentation Methods

### BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)

The default method. Uses RDKit's BRICS implementation to cut bonds at retrosynthetically sensible positions (amide bonds, amine-aryl bonds, ether bonds, etc.).

```
    BRICS fragmentation of a drug-like molecule:

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │    Ar─── NH ─── C(=O) ─── Ar ─── O ─── CH₂ ─── Ar │
    │              ↑           ↑          ↑               │
    │           BRICS        BRICS      BRICS             │
    │           bond         bond       bond              │
    └─────────────────────────────────────────────────────┘
                  │            │          │
                  ▼            ▼          ▼
         ┌────────────┐ ┌──────────┐ ┌────────┐
         │ [*]─NH─Ar  │ │ [*]─C(=O)│ │[*]─O─  │  ... etc.
         │  (1-AP)    │ │ ─Ar─[*]  │ │CH₂─Ar  │
         └────────────┘ │  (2-AP)  │ │ (2-AP) │
                        └──────────┘ └────────┘
```

**Two variants:**

#### Standard BRICS (`BRICSFragmenter`)
Cuts **all** BRICS bonds simultaneously, producing small fragments.

```python
from caveat.fragment import BRICSFragmenter
fragmenter = BRICSFragmenter(min_heavy_atoms=3)
fragments = fragmenter.fragment(mol)
```

#### Hierarchical BRICS (`HierarchicalBRICSFragmenter`)
Enumerates all combinations of 1, 2, ... N bond cuts. Produces fragments at **all size scales**.

```
    Hierarchical BRICS (max_cuts=2) on molecule with 4 BRICS bonds (a,b,c,d):

    1-cut combinations:          2-cut combinations:
    ─────────────────            ──────────────────
    cut {a}  → 2 frags          cut {a,b} → 3 frags
    cut {b}  → 2 frags          cut {a,c} → 3 frags
    cut {c}  → 2 frags          cut {a,d} → 3 frags
    cut {d}  → 2 frags          cut {b,c} → 3 frags
                                 cut {b,d} → 3 frags
    4 combinations               cut {c,d} → 3 frags
    → large fragments
    → 1-AP sidechains            6 combinations
                                 → medium fragments
                                 → 2-AP linkers
```

```python
from caveat.fragment import HierarchicalBRICSFragmenter
fragmenter = HierarchicalBRICSFragmenter(min_heavy_atoms=3, max_cuts=2)
fragments = fragmenter.fragment(mol)
```

- `max_cuts=1` → only single-bond cuts → large 1-AP fragments (sidechains)
- `max_cuts=2` → also 2-bond cuts → natural linker fragments for 2-AP queries
- Hierarchical is preferred for building comprehensive databases

**Build command:**
```bash
caveat build --input molecules.smi --db fragments.db --n-confs 5 --workers 8
```

### Reactive Handles (Building Blocks)

For **commercial building blocks** (e.g. from Mcule, Enamine, SpiroChem) that have explicit synthetic handles instead of BRICS-style cut points.

```
    Reactive handle → Attachment point conversion:

    Building block:              CAVEAT fragment:

       Br                          [*]
       │                           │
    ┌──┴──┐                     ┌──┴──┐
    │     │     ──────────►     │     │
    │ R   │     remove Br       │ R   │
    │     │     add [*]         │     │
    └─────┘                     └─────┘

    Keep-variant (preserves handle atom):

       NH₂                         NH─[*]
       │                           │
    ┌──┴──┐                     ┌──┴──┐
    │     │     ──────────►     │     │
    │ R   │     remove H        │ R   │
    │     │     add [*] to N    │     │
    └─────┘                     └─────┘
```

Converts reactive handles to attachment points:

| Handle | Conversion | Example |
|--------|-----------|---------|
| Br, Cl, I | R-X → R-[*] | ArBr → Ar-[*] |
| NH2 | R-NH2 → R-[*] | ArNH2 → Ar-[*] |
| B(OH)2 | R-B(OH)2 → R-[*] | ArB(OH)2 → Ar-[*] |
| COOH | R-COOH → R-[*] | ArCOOH → Ar-[*] |
| CHO | R-CHO → R-[*] | ArCHO → Ar-[*] |
| OH, SH | R-OH → R-[*] | ArOH → Ar-[*] |

**Keep variants** preserve part of the handle in the fragment, modeling reactions that retain atoms:

| Handle | Keep variant | Reaction modeled |
|--------|-------------|------------------|
| NH2 | R-NH2 → R-NH-[*] | Amide coupling (keeps NH) |
| OH | R-OH → R-O-[*] | Ether formation (keeps O) |
| COOH | R-COOH → R-CO-[*] | Amide from acid (keeps C=O) |

```python
from caveat.building_blocks import convert_building_block
fragments = convert_building_block(mol)  # returns list of Fragment objects
```

**Advantages over BRICS:**
- Fragments correspond to **purchasable compounds** with known synthetic routes
- Building blocks are directly orderable for synthesis
- Cover chemical space not well-represented in bioactivity databases

### C-H Bond Enumeration (Rigid Rings)

For **unfunctionalized ring systems** (pure carbon scaffolds with no reactive handles and no BRICS-cleavable bonds).

Enumerates **all pairs of C-H bonds** on the ring system as potential 2-AP attachment points:

```
    C-H pair enumeration on a bicyclic ring:

    Step 1: Identify all C-H bonds

            H   H                       1   2
            │   │                       │   │
        H── C ─ C ──H                  │   │
           ╱ ╲ ╱ ╲             ══►    3 ─ C ─ C ─ 4     (6 C-H bonds)
        H── C ─ C ──H                  │   │
            │   │                       5   6
            H   H

    Step 2: Enumerate all pairs → C(6,2) = 15 candidate 2-AP fragments

        Pair (1,4):                     Pair (2,5):

          [*]                              H
           │                               │
           C ─ C ── H                      C ─ C ──[*]
          ╱ ╲ ╱ ╲               vs.       ╱ ╲ ╱ ╲
       H── C ─ C ──[*]                [*]── C ─ C ── H
           │   │                           │   │
           H   H                           H   H

    Step 3: Screen pairs against query geometry (distance, angles, dihedral)

        ✗ Pair (1,2) — too close, wrong angle
        ✗ Pair (1,3) — wrong dihedral
        ✓ Pair (1,4) — geometric match!
        ✗ Pair (2,3) — too close
        ...
```

**Pipeline:**
1. Parse ring SMILES (e.g. from GDP, GDB-13 databases)
2. Enumerate all C-H pairs → candidate 2-AP fragments
3. Heuristic pre-filter: 1 conformer, generous tolerance (4x) → eliminates ~75% cheaply
4. Full screen: 10 conformers, standard tolerance (3x) → geometric ranking
5. Assembly + property calculation

**Advantages:**
- Accesses rigid, drug-like scaffolds that have zero rotatable bonds in the replacement
- Bicyclic and polycyclic ring systems can provide excellent geometric matches while dramatically reducing molecular flexibility
- Complementary to BRICS — these fragments can't be found by any bond-cutting method

**Limitations:**
- Combinatorial explosion: N carbons with H → O(N^2) pairs per ring
- Only applicable to ring systems with C-H bonds (not heteroatom rings)
- Fragments have no built-in heteroatom functionality

---

## Second-Order Extensions

For 2-AP fragments that are geometrically close but **too short** to span the full distance, small linker atoms can be inserted at one attachment point:

```
    Second-order extension:

    Original fragment (too short):

        [*]─── ring ───[*]
                              distance too small
                              ↕

    Extended fragment:

        [*]─── ring ───CH₂───[*]       (methylene linker)
        [*]─── ring ───NH────[*]        (amine linker)
        [*]─── ring ───O─────[*]        (ether linker)
        [*]─── ring ───CH₂CH₂─[*]      (ethylene linker)

    Linker library: CH₂, NH, O, CH₂CH₂, CH₂O, OCH₂, CH₂NH, NHCH₂
```

- Dramatically improves out-of-plane geometry (linkers add conformational flexibility at junctions)
- Expands the hit space for fragments that would otherwise be rejected

---

## Choosing a Strategy

| Goal | Mode | Fragmentation | Notes |
|------|------|--------------|-------|
| Replace a linker | 2-AP | BRICS / Hierarchical | Standard approach |
| Replace a sidechain | 1-AP | BRICS + pharmacophore | Add pharmacophore constraint for selectivity |
| Find purchasable replacements | 2-AP or 1-AP | Reactive handles | Mcule, SpiroChem, Enamine catalogs |
| Reduce flexibility | 2-AP | C-H enumeration | Rigid polycyclic rings, RotBonds typically decreases |
| Maximize chemical diversity | Combination | Mix 2-AP + 1-AP | Combinatorial assembly of linker + cap |
| Second-order extensions | 2-AP | BRICS + linker insertion | Extend short fragments with small linkers (CH2, NH, O) |

---

## Database Sources

### Fragment Counts

ChEMBL 35 (2.2M drug-like molecules) produces **16M total fragments** via standard BRICS, or **2.3M** via hierarchical BRICS (max 2 cuts). The numbers below are for the **embedded screen databases** — subsets filtered and embedded with 3D conformers:

| Database | Fragmentation | Total frags | 1-AP | 2-AP | Conformers | Size | Filters |
|----------|--------------|-------------|------|------|------------|------|---------|
| ChEMBL hierarchical screen | Hierarchical BRICS | 109K | 48K | 61K | 244K | 142 MB | HA ≤ 14, 1-2 AP, ArRings ≤ 1, RotBonds ≤ 2 |
| ChEMBL RotBonds-1 screen | Standard BRICS | 254K | 115K | 104K | 706K | 593 MB | 1-3 AP, RotBonds ≤ 1 |
| ChEMBL RotBonds-2 screen | Standard BRICS | 182K | — | 182K | 182K | 210 MB | HA ≤ 14, 2 AP only, non-aromatic, RotBonds ≤ 2 |
| Approved drugs screen | BRICS | 8.5K | 4.1K | 4.4K | 8.5K | 8 MB | HA ≤ 14, 1-2 AP, ArRings ≤ 1 |
| Mcule building blocks | Reactive handles | 91K | 78K | 11K | 250K | 170 MB | All converted handles (HA ≤ 14 from BB size) |
| Mcule keep-variants | Reactive handles | 52K | 46K | 5.9K | 179K | 110 MB | Keep-variant handles only |
| SpiroChem building blocks | Reactive handles | 4.2K | 3.1K | 1.0K | 11K | 7 MB | All converted handles |
| GDP/GDB-13 rings | C-H enumeration | — | — | — | — | On-the-fly | Pure carbon rings, 5-10 atoms |

### Unembedded Fragments

The full ChEMBL BRICS fragmentation produces **16M fragments** (7.1M 1-AP, 6.3M 2-AP, 2.7M 3+ AP), of which only ~700K are currently embedded with 3D conformers. The hierarchical BRICS database has **2.3M fragments** (966K 1-AP, 1.35M 2-AP), also mostly unembedded.

### Conformer Generation Cost

Embedding a single fragment with 5 conformers takes **~0.5 seconds** on average (varies with fragment size and ring complexity). At this rate:

- **700K fragments** (current screen DBs): ~4 days on 1 CPU, ~12 hours on 8 CPUs
- **2.3M fragments** (full hierarchical): ~13 days on 1 CPU, ~2 days on 8 CPUs
- **16M fragments** (full BRICS): ~93 days on 1 CPU, ~12 days on 8 CPUs

A fully embedded database of all 16M ChEMBL fragments at 5 conformers each would be approximately **50-60 GB**. In practice, pre-filtering by heavy atom count, rotatable bonds, or aromaticity reduces this by 5-10x.

### Conformers Per Fragment

5 conformers per fragment provides **3.2x more hits** than a single conformer (tested on a 2-AP linker replacement query), with the best geometric score improving from 0.44 to 0.17. The gains diminish beyond 5-10 conformers for most fragment sizes.
