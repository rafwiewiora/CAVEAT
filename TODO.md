# CAVEAT TODO

## Documentation & Tutorials

- [ ] **Interactive tutorial/explainer** — web-based tool where users can:
  - Try example molecules (paste SMILES or pick from presets)
  - Click through each pipeline step with visualizations
  - "Learn more" expandable sections with textbook-style explanations of the math/code:
    - BRICS bond identification and fragmentation rules
    - 4D geometric descriptor computation (distance, angles, dihedral)
    - KD-tree construction and nearest-neighbor search
    - Rodrigues' rotation formula (1-AP alignment)
    - Kabsch/SVD algorithm (2-AP alignment)
    - MMFF junction optimization and planarity enforcement
  - Interactive 3D viewer (e.g. 3Dmol.js) showing parent, fragment, and assembled molecules
  - Side-by-side comparison of original substructure vs replacement

## Database & Scaling

- [ ] **Full ChEMBL embedding** — embed all 2.3M hierarchical fragments (currently ~700K embedded across screen DBs)
- [ ] **DB-level deduplication** — strip BRICS isotope labels before canonicalizing at build time (would reduce fragments ~31%)
- [ ] **Fragment-level property pre-filtering** — compute delta properties on fragments themselves to enable pre-screening before assembly
- [ ] **Source tracking** — populate `sources` table in filtered screen DBs (currently empty after filtering)

## Screening & Analysis

- [ ] **Privileged fragment scoring** — join fragment sources with ChEMBL bioactivity/ADMET data to prioritize fragments from active compounds
- [ ] **Matched Molecular Pairs (MMP)** — extract MMPs from ChEMBL to identify activity cliffs and inform fragment selection
- [ ] **Synthetic accessibility scoring** — integrate SA score or building block availability into fragment ranking

## Code Quality

- [ ] **CLI for 1-AP pharmacophore screen** — currently script-only, add `caveat query --pharmacophore` option
- [ ] **CLI for combination mode** — `caveat combine` command for 2-AP x 1-AP and 2-AP x 2-AP assembly
- [ ] **Streaming assembly** — for million-scale results, write assembled molecules incrementally instead of holding all in memory
