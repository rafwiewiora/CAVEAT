"""Build a self-contained HTML page with Mol* 3D viewer for CAVEAT results.

Shows assembled products alongside their source approved drugs,
with drug name, indication, and property information.

Usage:
    python scripts/make_molstar_html.py \
        --results-dir results_tyrout_approved2
"""

import argparse
import csv
import json
import os
import html as html_mod

from rdkit import Chem


def sdf_to_string(path):
    """Read an SDF file and return its content as a string."""
    with open(path) as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = args.results_dir

    # Load drug info
    drug_info_path = os.path.join(results_dir, "drug_info.json")
    with open(drug_info_path) as f:
        drug_info = json.load(f)

    # Load properties CSV
    props = []
    with open(os.path.join(results_dir, "properties.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            props.append(row)

    parent_row = props[0]  # rank=0
    hit_rows = [r for r in props if r["rank"] != "0"]

    # Read parent SDF
    parent_sdf = sdf_to_string(os.path.join(results_dir, "parent.sdf"))

    # Build entries for each hit
    entries = []
    for row in hit_rows:
        rank = int(row["rank"])
        frag_smi = row["fragment_smiles"]
        assembled_smi = row["assembled_smiles"]
        geo_score = row["geo_score"]

        assembled_path = os.path.join(results_dir, "assembled", f"assembled_{rank:03d}.sdf")
        frag_path = os.path.join(results_dir, "fragments", f"frag_{rank:03d}.sdf")
        drug_path = os.path.join(results_dir, "source_drugs", f"drug_{rank:03d}.sdf")

        assembled_sdf = sdf_to_string(assembled_path) if os.path.exists(assembled_path) else None
        frag_sdf = sdf_to_string(frag_path) if os.path.exists(frag_path) else None

        # Get drug info
        drug_sdf = None
        chembl_id = None
        drug_name = None
        indication = None
        drug_smi = None

        if os.path.exists(drug_path):
            drug_sdf = sdf_to_string(drug_path)
            suppl = Chem.SDMolSupplier(drug_path)
            for mol in suppl:
                if mol:
                    if mol.HasProp("chembl_id"):
                        chembl_id = mol.GetProp("chembl_id")
                    if mol.HasProp("source_smiles"):
                        drug_smi = mol.GetProp("source_smiles")
                break

        if chembl_id and chembl_id in drug_info:
            info = drug_info[chembl_id]
            drug_name = info.get("name")
            indication = info.get("indication")

        entries.append({
            "rank": rank,
            "frag_smi": frag_smi,
            "assembled_smi": assembled_smi,
            "geo_score": geo_score,
            "mw": row["MW"],
            "clogp": row["cLogP"],
            "hba": row["HBA"],
            "hbd": row["HBD"],
            "rotbonds": row["RotBonds"],
            "tpsa": row["TPSA"],
            "dmw": row.get("dMW", ""),
            "dclogp": row.get("dcLogP", ""),
            "assembled_sdf": assembled_sdf,
            "frag_sdf": frag_sdf,
            "drug_sdf": drug_sdf,
            "chembl_id": chembl_id,
            "drug_name": drug_name,
            "indication": indication,
            "drug_smi": drug_smi,
        })

    # Build HTML
    html = build_html(parent_row, parent_sdf, entries)

    out_path = os.path.join(results_dir, "viewer.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Wrote {out_path}")
    print(f"Open in browser: file://{os.path.abspath(out_path)}")


def build_html(parent_row, parent_sdf, entries):
    # Serialize SDF data as JSON for JavaScript
    sdf_data = {
        "parent": parent_sdf,
    }
    for e in entries:
        r = e["rank"]
        if e["assembled_sdf"]:
            sdf_data[f"assembled_{r}"] = e["assembled_sdf"]
        if e["frag_sdf"]:
            sdf_data[f"frag_{r}"] = e["frag_sdf"]
        if e["drug_sdf"]:
            sdf_data[f"drug_{r}"] = e["drug_sdf"]

    # Build card HTML for each entry
    cards_html = ""
    for e in entries:
        r = e["rank"]
        name = e["drug_name"] or "Unknown"
        chembl = e["chembl_id"] or ""
        indication = e["indication"] or "N/A"
        score = e["geo_score"]

        chembl_link = f'<a href="https://www.ebi.ac.uk/chembl/compound_report_card/{chembl}/" target="_blank">{chembl}</a>' if chembl else ""

        dmw = e["dmw"]
        dclogp = e["dclogp"]
        delta_str = ""
        if dmw:
            delta_str = f"&Delta;MW {float(dmw):+.0f}, &Delta;cLogP {float(dclogp):+.1f}"

        cards_html += f"""
        <div class="card" data-rank="{r}" onclick="selectEntry({r})">
            <div class="card-rank">#{r}</div>
            <div class="card-body">
                <div class="drug-name">{html_mod.escape(name)}</div>
                <div class="indication">{html_mod.escape(indication)}</div>
                <div class="chembl-id">{chembl_link}</div>
                <div class="frag-smi"><code>{html_mod.escape(e['frag_smi'])}</code></div>
                <div class="props">
                    MW {e['mw']} | cLogP {e['clogp']} | HBA {e['hba']} | HBD {e['hbd']} | RotB {e['rotbonds']} | TPSA {e['tpsa']}
                </div>
                <div class="delta">{delta_str}</div>
                <div class="score">Geo score: {score}</div>
            </div>
        </div>
"""

    parent_props = f"MW {parent_row['MW']} | cLogP {parent_row['cLogP']} | HBA {parent_row['HBA']} | HBD {parent_row['HBD']} | RotB {parent_row['RotBonds']} | TPSA {parent_row['TPSA']}"

    # Pre-build JSON strings to avoid f-string escaping issues
    sdf_data_json = json.dumps(sdf_data)
    entries_json = json.dumps([{
        "rank": e["rank"],
        "drug_name": e["drug_name"],
        "chembl_id": e["chembl_id"],
        "indication": e["indication"],
        "frag_smi": e["frag_smi"],
    } for e in entries])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAVEAT Fragment Replacement Results</title>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; height: 100vh; overflow: hidden; }}

.layout {{ display: grid; grid-template-columns: 380px 1fr; grid-template-rows: auto 1fr; height: 100vh; }}

.header {{
    grid-column: 1 / -1;
    background: #1a1d27;
    padding: 12px 24px;
    border-bottom: 1px solid #2a2d3a;
    display: flex;
    align-items: center;
    gap: 24px;
}}
.header h1 {{ font-size: 18px; font-weight: 600; color: #fff; }}
.header .parent-info {{ font-size: 13px; color: #8b8fa3; }}
.header .controls {{ margin-left: auto; display: flex; gap: 8px; }}
.header .controls button {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid #3a3d4a;
    background: #1a1d27; color: #c0c4d6; cursor: pointer; font-size: 12px;
    transition: all 0.15s;
}}
.header .controls button:hover {{ background: #2a2d3a; color: #fff; }}
.header .controls button.active {{ background: #4a6cf7; border-color: #4a6cf7; color: #fff; }}

.sidebar {{
    background: #14161e;
    overflow-y: auto;
    border-right: 1px solid #2a2d3a;
    padding: 8px;
}}

.card {{
    padding: 12px;
    margin-bottom: 6px;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s;
}}
.card:hover {{ background: #1e2130; border-color: #3a3d4a; }}
.card.selected {{ background: #1a2744; border-color: #4a6cf7; }}
.card-rank {{ font-size: 11px; color: #5a5e72; font-weight: 600; margin-bottom: 4px; }}
.drug-name {{ font-size: 14px; font-weight: 600; color: #e8eaf0; margin-bottom: 2px; }}
.indication {{ font-size: 12px; color: #7c8db5; margin-bottom: 4px; font-style: italic; }}
.chembl-id {{ font-size: 11px; margin-bottom: 4px; }}
.chembl-id a {{ color: #6b8aed; text-decoration: none; }}
.chembl-id a:hover {{ text-decoration: underline; }}
.frag-smi {{ font-size: 11px; color: #6b7080; margin-bottom: 4px; }}
.frag-smi code {{ background: #1a1d27; padding: 2px 5px; border-radius: 3px; }}
.props {{ font-size: 11px; color: #6b7080; }}
.delta {{ font-size: 11px; color: #8b9dc3; margin-top: 2px; }}
.score {{ font-size: 11px; color: #5a5e72; margin-top: 2px; }}

.viewer-container {{
    position: relative;
    background: #1a1d27;
}}
#viewer {{
    width: 100%;
    height: 100%;
}}

.viewer-legend {{
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(20, 22, 30, 0.85);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 12px;
    pointer-events: none;
    backdrop-filter: blur(8px);
}}
.legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

.viewer-info {{
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(20, 22, 30, 0.85);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    backdrop-filter: blur(8px);
    max-width: 60%;
}}
.viewer-info .vi-name {{ font-weight: 600; color: #fff; font-size: 15px; }}
.viewer-info .vi-indication {{ color: #7c8db5; font-style: italic; margin-top: 2px; }}
.viewer-info .vi-chembl {{ color: #6b8aed; margin-top: 2px; }}
.viewer-info .vi-chembl a {{ color: #6b8aed; text-decoration: none; }}
</style>
</head>
<body>
<div class="layout">
    <div class="header">
        <h1>CAVEAT Fragment Replacements</h1>
        <div class="parent-info">Parent: {parent_props}</div>
        <div class="controls">
            <button id="btn-assembled" class="active" onclick="toggleLayer('assembled')">Assembled</button>
            <button id="btn-fragment" onclick="toggleLayer('fragment')">Fragment</button>
            <button id="btn-drug" onclick="toggleLayer('drug')">Source Drug</button>
            <button id="btn-parent" class="active" onclick="toggleLayer('parent')">Parent</button>
        </div>
    </div>
    <div class="sidebar" id="sidebar">
{cards_html}
    </div>
    <div class="viewer-container">
        <div id="viewer"></div>
        <div class="viewer-legend">
            <div class="legend-item"><div class="legend-dot" style="background:#9e9e9e"></div> Parent (C)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#58c4dd"></div> Assembled (C)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#e07060"></div> Source Drug (C)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#50c878"></div> Fragment (C)</div>
        </div>
        <div class="viewer-info" id="viewer-info" style="display:none"></div>
    </div>
</div>

<script>
const SDF_DATA = {sdf_data_json};
const ENTRIES = {entries_json};

let stage;
let currentRank = null;
let layers = {{ parent: true, assembled: true, fragment: false, drug: false }};
let components = {{}};

// Custom color scheme: element colors with custom carbon
function makeScheme(carbonColor) {{
    return NGL.ColormakerRegistry.addScheme(function(params) {{
        this.atomColor = function(atom) {{
            switch(atom.element) {{
                case 'C': return carbonColor;
                case 'N': return 0x4060CF;
                case 'O': return 0xE04040;
                case 'S': return 0xD0C030;
                case 'F': return 0x70D040;
                case 'CL': return 0x30C030;
                case 'BR': return 0xA04040;
                case 'P': return 0xE08000;
                case 'H': return (carbonColor & 0xfefefe) >> 1 | 0x808080;
                default: return 0xCCCCCC;
            }}
        }};
    }});
}}

const parentScheme = makeScheme(0xA0A0A0);
const assembledScheme = makeScheme(0x50B8D8);
const fragmentScheme = makeScheme(0x45C07A);
const drugScheme = makeScheme(0xD07060);

function initViewer() {{
    stage = new NGL.Stage('viewer', {{
        backgroundColor: '#1a1d27',
        ambientColor: 0x444466,
        ambientIntensity: 0.3,
        quality: 'high',
    }});

    if (ENTRIES.length > 0) {{
        selectEntry(ENTRIES[0].rank);
    }}
}}

async function loadSDF(key, name, scheme, scale) {{
    if (!SDF_DATA[key]) return null;
    const blob = new Blob([SDF_DATA[key]], {{type: 'text/plain'}});
    const comp = await stage.loadFile(blob, {{ext: 'sdf', name: name}});
    comp.addRepresentation('ball+stick', {{
        colorScheme: scheme,
        aspectRatio: 1.8,
        radiusSize: 0.15 * scale,
        multipleBond: 'symmetric',
        sphereDetail: 2,
    }});
    return comp;
}}

async function selectEntry(rank) {{
    document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
    const card = document.querySelector(`.card[data-rank="${{rank}}"]`);
    if (card) {{
        card.classList.add('selected');
        card.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}
    currentRank = rank;
    await updateViewer();
    updateInfoPanel(rank);
}}

async function updateViewer() {{
    stage.removeAllComponents();
    components = {{}};

    if (layers.parent && SDF_DATA.parent) {{
        components.parent = await loadSDF('parent', 'parent', parentScheme, 0.9);
    }}
    if (layers.assembled && currentRank && SDF_DATA['assembled_' + currentRank]) {{
        components.assembled = await loadSDF('assembled_' + currentRank, 'assembled', assembledScheme, 1.0);
    }}
    if (layers.fragment && currentRank && SDF_DATA['frag_' + currentRank]) {{
        components.fragment = await loadSDF('frag_' + currentRank, 'fragment', fragmentScheme, 1.1);
    }}
    if (layers.drug && currentRank && SDF_DATA['drug_' + currentRank]) {{
        components.drug = await loadSDF('drug_' + currentRank, 'drug', drugScheme, 0.95);
    }}

    stage.autoView(800);
}}

function toggleLayer(name) {{
    layers[name] = !layers[name];
    document.getElementById('btn-' + name).classList.toggle('active');
    updateViewer();
}}

function updateInfoPanel(rank) {{
    const entry = ENTRIES.find(e => e.rank === rank);
    if (!entry) return;
    const panel = document.getElementById('viewer-info');
    panel.style.display = 'block';
    let h = '';
    if (entry.drug_name) h += `<div class="vi-name">${{entry.drug_name}}</div>`;
    if (entry.indication) h += `<div class="vi-indication">${{entry.indication}}</div>`;
    if (entry.chembl_id) h += `<div class="vi-chembl"><a href="https://www.ebi.ac.uk/chembl/compound_report_card/${{entry.chembl_id}}/" target="_blank">${{entry.chembl_id}}</a></div>`;
    panel.innerHTML = h;
}}

document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowDown' || e.key === 'j') {{
        e.preventDefault();
        const idx = ENTRIES.findIndex(en => en.rank === currentRank);
        if (idx < ENTRIES.length - 1) selectEntry(ENTRIES[idx + 1].rank);
    }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
        e.preventDefault();
        const idx = ENTRIES.findIndex(en => en.rank === currentRank);
        if (idx > 0) selectEntry(ENTRIES[idx - 1].rank);
    }} else if (e.key === '1') {{ toggleLayer('assembled');
    }} else if (e.key === '2') {{ toggleLayer('fragment');
    }} else if (e.key === '3') {{ toggleLayer('drug');
    }} else if (e.key === '4') {{ toggleLayer('parent');
    }}
}});

window.addEventListener('load', initViewer);
window.addEventListener('resize', () => {{ if (stage) stage.handleResize(); }});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
