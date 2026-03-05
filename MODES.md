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

ChEMBL 35 (2.2M drug-like molecules) produces **16M total fragments** via standard BRICS, or **2.3M** via hierarchical BRICS (max 2 cuts). The numbers below are for the **embedded screen databases** — subsets filtered by heavy atom count, rotatable bonds, and/or aromaticity, then embedded with 3D conformers:

| Database | Fragmentation | Total frags | 1-AP | 2-AP | Conformers | Size |
|----------|--------------|-------------|------|------|------------|------|
| ChEMBL hierarchical screen | Hierarchical BRICS | 109K | 48K | 61K | 244K | 142 MB |
| ChEMBL full RotBonds-1 screen | Standard BRICS | 254K | 115K | 104K | 706K | 593 MB |
| ChEMBL RotBonds-2 screen | Standard BRICS | 182K | — | 182K | 182K | 210 MB |
| Approved drugs screen | BRICS | 8.5K | 4.1K | 4.4K | 8.5K | 8 MB |
| Mcule building blocks | Reactive handles | 91K | 78K | 11K | 250K | 170 MB |
| Mcule keep-variants | Reactive handles | 52K | 46K | 5.9K | 179K | 110 MB |
| SpiroChem building blocks | Reactive handles | 4.2K | 3.1K | 1.0K | 11K | 7 MB |
| GDP/GDB-13 rings | C-H enumeration | — | — | — | — | On-the-fly |

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
