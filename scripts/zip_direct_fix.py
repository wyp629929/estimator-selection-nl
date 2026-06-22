"""
===========================================================================
  ZIP-direct fix — 完全绕过 python-docx save 问题
  - 直接操作 main.docx 的 ZIP 包内的 document.xml
  - 使用 lxml 修改 XML，而不依赖 python-docx 保存
  - 仅修改格式和表格，不重写正文
===========================================================================
"""

import zipfile, shutil, os, sys, re
from lxml import etree
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex_to_omml import build_cell_paragraph

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

SOURCE = "/Users/wangyaoping/ml-inference-paper/paper/main.docx"
OUTPUT = "/Users/wangyaoping/Desktop/paper_final.docx"
FONT_NAME = "Times New Roman"


def ns(prefix):
    """Namespace lookup."""
    return {
        "w": W_NS,
        "m": OMML_NS,
        "r": R_NS,
    }[prefix]


def qn(tag):
    """Qualified name with namespace."""
    prefix, local = tag.split(":")
    return f"{{{ns(prefix)}}}{local}"


# ── Verified table data ──

TABLE1_DATA = [
    ("Linear",          "$y = X\\beta + \\varepsilon$, $\\beta = (2, -1.5, 0.8, 0, 0)$", "Baseline: OLS optimal"),
    ("Semiparametric",  "$y = X\\beta + 0.3\\sin(X_1) + \\varepsilon$",                   "Mild misspecification"),
    ("Nonlinear",       "$y = \\sin X_1 + \\log(1+|X_2|) + X_3 X_4 + \\varepsilon$",      "Strong nonlinearity"),
    ("High-dim sparse", "$p=100$, 5 non-zero coefficients",                                "Selection + estimation"),
]

TABLE2_MSE = [
    ("highdim", "dnn",      "3.3050", "0.6942"),
    ("highdim", "lasso",    "1.0735", "0.1510"),
    ("highdim", "lightgbm", "3.5685", "0.8672"),
    ("highdim", "mlp",      "7.9618", "1.8603"),
    ("highdim", "ols",      "1.3985", "0.1913"),
    ("highdim", "rf",       "7.1531", "2.0531"),
    ("highdim", "ridge",    "1.3938", "0.1902"),
    ("highdim", "xgboost",  "6.3345", "1.7439"),
    ("linear", "dnn",       "1.2156", "0.1572"),
    ("linear", "lasso",     "1.0516", "0.1234"),
    ("linear", "lightgbm",  "1.6059", "0.2051"),
    ("linear", "mlp",       "1.1082", "0.1286"),
    ("linear", "ols",       "1.0336", "0.1224"),
    ("linear", "rf",        "1.5009", "0.2001"),
    ("linear", "ridge",     "1.0333", "0.1224"),
    ("linear", "xgboost",   "1.6309", "0.2179"),
    ("nonlinear", "dnn",    "1.3440", "0.1551"),
    ("nonlinear", "lasso",  "2.1548", "0.3204"),
    ("nonlinear", "lightgbm","1.5530", "0.2196"),
    ("nonlinear", "mlp",    "1.2196", "0.1528"),
    ("nonlinear", "ols",    "2.1912", "0.3295"),
    ("nonlinear", "rf",     "1.7650", "0.3085"),
    ("nonlinear", "ridge",  "2.1908", "0.3293"),
    ("nonlinear", "xgboost","1.7451", "0.2731"),
    ("semiparametric", "dnn",    "1.2080", "0.1382"),
    ("semiparametric", "lasso",  "1.0555", "0.1270"),
    ("semiparametric", "lightgbm","1.6102", "0.1947"),
    ("semiparametric", "mlp",    "1.0853", "0.1223"),
    ("semiparametric", "ols",    "1.0232", "0.1142"),
    ("semiparametric", "rf",     "1.5374", "0.2077"),
    ("semiparametric", "ridge",  "1.0234", "0.1145"),
    ("semiparametric", "xgboost","1.6723", "0.2090"),
]


# ══════════════════════════════════════════════════════════════
# XML helpers
# ══════════════════════════════════════════════════════════════

def make_spacing_element():
    """w:spacing with line=240 atLeast, before=0, after=0."""
    sp = etree.SubElement(etree.Element("_dummy"), qn("w:spacing"))
    sp.set(qn("w:line"), "240")
    sp.set(qn("w:lineRule"), "atLeast")
    sp.set(qn("w:before"), "0")
    sp.set(qn("w:after"), "0")
    return sp


def set_or_add_spacing(p_element):
    """Add or update w:spacing in w:pPr."""
    pPr = p_element.find(qn("w:pPr"))
    if pPr is None:
        pPr = etree.SubElement(p_element, qn("w:pPr"))
        p_element.insert(0, pPr)

    spacing = pPr.find(qn("w:spacing"))
    if spacing is None:
        spacing = etree.SubElement(pPr, qn("w:spacing"))
    spacing.set(qn("w:line"), "240")
    spacing.set(qn("w:lineRule"), "atLeast")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")


def set_run_font(r_element):
    """Force TNR 20 half-pt on a w:r element."""
    rPr = r_element.find(qn("w:rPr"))
    if rPr is None:
        rPr = etree.SubElement(r_element, qn("w:rPr"))
        r_element.insert(0, rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = etree.SubElement(rPr, qn("w:rFonts"))
    rFonts.set(qn("w:ascii"), FONT_NAME)
    rFonts.set(qn("w:hAnsi"), FONT_NAME)
    rFonts.set(qn("w:eastAsia"), FONT_NAME)
    sz = rPr.find(qn("w:sz"))
    if sz is None:
        sz = etree.SubElement(rPr, qn("w:sz"))
    sz.set(qn("w:val"), "20")
    szCs = rPr.find(qn("w:szCs"))
    if szCs is None:
        szCs = etree.SubElement(rPr, qn("w:szCs"))
    szCs.set(qn("w:val"), "20")


def make_text_run(text, bold=False, align_val="left"):
    """Create w:p containing a single w:r with formatted text."""
    p = etree.Element(qn("w:p"))
    pPr = etree.SubElement(p, qn("w:pPr"))
    jc = etree.SubElement(pPr, qn("w:jc"))
    jc.set(qn("w:val"), align_val)
    set_or_add_spacing(p)

    r = etree.SubElement(p, qn("w:r"))
    rPr = etree.SubElement(r, qn("w:rPr"))
    rFonts = etree.SubElement(rPr, qn("w:rFonts"))
    rFonts.set(qn("w:ascii"), FONT_NAME)
    rFonts.set(qn("w:hAnsi"), FONT_NAME)
    sz = etree.SubElement(rPr, qn("w:sz"))
    sz.set(qn("w:val"), "20")
    szCs = etree.SubElement(rPr, qn("w:szCs"))
    szCs.set(qn("w:val"), "20")
    if bold:
        etree.SubElement(rPr, qn("w:b"))
    t = etree.SubElement(r, qn("w:t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p


def make_omml_paragraph(text, align_val="left"):
    """Build paragraph with $...$ converted to OMML."""
    # First check if there's any $...$ content
    if "$" not in text:
        return make_text_run(text, align_val=align_val)
    # Use the build_cell_paragraph from latex_to_omml
    new_p = build_cell_paragraph(text, align_val=align_val)
    # Update font sizes
    for r in new_p.iter(qn("w:r")):
        rPr = r.find(qn("w:rPr"))
        if rPr is None:
            rPr = etree.SubElement(r, qn("w:rPr"))
            r.insert(0, rPr)
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = etree.SubElement(rPr, qn("w:rFonts"))
        rFonts.set(qn("w:ascii"), FONT_NAME)
        rFonts.set(qn("w:hAnsi"), FONT_NAME)
        sz = rPr.find(qn("w:sz"))
        if sz is None:
            sz = etree.SubElement(rPr, qn("w:sz"))
        sz.set(qn("w:val"), "20")
        szCs = rPr.find(qn("w:szCs"))
        if szCs is None:
            szCs = etree.SubElement(rPr, qn("w:szCs"))
        szCs.set(qn("w:val"), "20")
    set_or_add_spacing(new_p)
    return new_p


def apply_three_line_borders(tbl):
    """Set three-line borders on a w:tbl element at XML level."""
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = etree.Element(qn("w:tblPr"))
        tbl.insert(0, tblPr)

    borders = etree.SubElement(tblPr, qn("w:tblBorders")) if tblPr.find(qn("w:tblBorders")) is None else tblPr.find(qn("w:tblBorders"))
    # Remove all existing border children
    for child in list(borders):
        borders.remove(child)

    border_specs = [
        ("top", "single", "12", "0", "000000"),
        ("left", "none", "0", "0", "auto"),
        ("right", "none", "0", "0", "auto"),
        ("insideV", "none", "0", "0", "auto"),
        ("insideH", "none", "0", "0", "auto"),
        ("bottom", "single", "12", "0", "000000"),
    ]
    for name, val, sz, space, color in border_specs:
        b = etree.SubElement(borders, qn(f"w:{name}"))
        b.set(qn("w:val"), val)
        b.set(qn("w:sz"), sz)
        b.set(qn("w:space"), space)
        b.set(qn("w:color"), color)


def set_header_bottom_border(tc):
    """Add 0.75pt bottom border to header cell."""
    tcPr = tc.find(qn("w:tcPr"))
    if tcPr is None:
        tcPr = etree.SubElement(tc, qn("w:tcPr"))
        tc.insert(0, tcPr)
    tcBorders = etree.SubElement(tcPr, qn("w:tcBorders"))
    bottom = etree.SubElement(tcBorders, qn("w:bottom"))
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")


def set_cell_padding(tc, top=40, bottom=40, left=80, right=80):
    """Set cell padding in twips (default 2pt top/bottom, 4pt left/right)."""
    tcPr = tc.find(qn("w:tcPr"))
    if tcPr is None:
        tcPr = etree.SubElement(tc, qn("w:tcPr"))
        tc.insert(0, tcPr)
    # Remove existing tcMar
    existing_mar = tcPr.find(qn("w:tcMar"))
    if existing_mar is not None:
        tcPr.remove(existing_mar)
    tcMar = etree.SubElement(tcPr, qn("w:tcMar"))
    for side, w in [("top", str(top)), ("bottom", str(bottom)), ("left", str(left)), ("right", str(right))]:
        m = etree.SubElement(tcMar, qn(f"w:{side}"))
        m.set(qn("w:w"), w)
        m.set(qn("w:type"), "dxa")


def make_table1():
    """Build Table 1 (Simulation scenarios) as w:tbl XML element."""
    tbl = etree.Element(qn("w:tbl"))
    apply_three_line_borders(tbl)

    headers = ["Scenario", "Data generating process", "Purpose"]
    data = TABLE1_DATA

    # Header row
    tr_h = etree.SubElement(tbl, qn("w:tr"))
    for h in headers:
        tc = etree.SubElement(tr_h, qn("w:tc"))
        tc.append(make_text_run(h, bold=True, align_val="center"))
        set_header_bottom_border(tc)
        set_cell_padding(tc)

    # Data rows
    for scenario, dgp, purpose in data:
        tr = etree.SubElement(tbl, qn("w:tr"))
        # Column 0: scenario name
        tc0 = etree.SubElement(tr, qn("w:tc"))
        tc0.append(make_text_run(scenario, align_val="left"))
        set_cell_padding(tc0)

        # Column 1: DGP with OMML
        tc1 = etree.SubElement(tr, qn("w:tc"))
        tc1.append(make_omml_paragraph(dgp, align_val="left"))
        set_cell_padding(tc1)

        # Column 2: purpose
        tc2 = etree.SubElement(tr, qn("w:tc"))
        tc2.append(make_text_run(purpose, align_val="left"))
        set_cell_padding(tc2)

    return tbl


def make_table2():
    """Build Table 2 (Test MSE) with vertical merge."""
    tbl = etree.Element(qn("w:tbl"))
    apply_three_line_borders(tbl)

    headers = ["Scenario", "Method", "MSE", "SD"]
    data = TABLE2_MSE

    # Header row
    tr_h = etree.SubElement(tbl, qn("w:tr"))
    for h in headers:
        tc = etree.SubElement(tr_h, qn("w:tc"))
        tc.append(make_text_run(h, bold=True, align_val="center"))
        set_header_bottom_border(tc)
        set_cell_padding(tc)

    # Data rows with vertical merge
    group_size = 8
    n_groups = len(data) // group_size

    for g in range(n_groups):
        start = g * group_size
        for ri in range(start, start + group_size):
            scenario, method, mse, sd = data[ri]
            tr = etree.SubElement(tbl, qn("w:tr"))

            # Column 0: scenario (vertically merged)
            tc0 = etree.SubElement(tr, qn("w:tc"))
            tcPr0 = etree.SubElement(tc0, qn("w:tcPr"))
            tc0.insert(0, tcPr0)
            vm = etree.SubElement(tcPr0, qn("w:vMerge"))
            if ri == start:
                vm.set(qn("w:val"), "restart")
                va = etree.SubElement(tcPr0, qn("w:vAlign"))
                va.set(qn("w:val"), "center")
                tc0.append(make_text_run(scenario, align_val="center"))
            else:
                vm.set(qn("w:val"), "continue")
                # Empty cell
                tc0.append(etree.Element(qn("w:p")))
            set_cell_padding(tc0)

            # Columns 1-3
            for ci, val in enumerate([method, mse, sd]):
                tc = etree.SubElement(tr, qn("w:tc"))
                tc.append(make_text_run(val, align_val="center" if ci > 0 else "left"))
                set_cell_padding(tc)

            # Column widths via tcPr
            widths_twips = [1872, 1584, 1440, 1440]  # Inches to twips: 1.3, 1.1, 1.0, 1.0
            for ci, tc in enumerate(tr.findall(qn("w:tc"))):
                tcPr = tc.find(qn("w:tcPr"))
                if tcPr is None:
                    tcPr = etree.SubElement(tc, qn("w:tcPr"))
                    tc.insert(0, tcPr)
                tcW = etree.SubElement(tcPr, qn("w:tcW"))
                tcW.set(qn("w:w"), str(widths_twips[ci]))
                tcW.set(qn("w:type"), "dxa")

    return tbl


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("Reading main.docx...")
    # Read all files from source zip
    with zipfile.ZipFile(SOURCE, 'r') as z_in:
        all_files = {}
        for name in z_in.namelist():
            all_files[name] = z_in.read(name)

    # Parse document.xml
    doc_xml = all_files["word/document.xml"]
    root = etree.fromstring(doc_xml)
    body = root.find(qn("w:body"))

    print(f"Processing {len(body)} body children...")

    # ── Step 1: Fix all paragraph formatting ──
    para_count = 0
    for child in body:
        if child.tag == qn("w:p"):
            # Add/update spacing
            set_or_add_spacing(child)
            # Fix font on all runs
            for r in child.iter(qn("w:r")):
                set_run_font(r)
            # Also fix OMML runs
            for r in child.iter(qn("m:r")):
                rPr = r.find(qn("m:rPr"))
                if rPr is not None:
                    rFonts = rPr.find(qn("m:rFonts"))
                    if rFonts is not None:
                        rFonts.set(qn("m:ascii"), FONT_NAME)
                        rFonts.set(qn("m:hAnsi"), FONT_NAME)
                    rFonts_old = rPr.find(qn("w:rFonts"))
                    if rFonts_old is not None:
                        rPr.remove(rFonts_old)
                    # Also check for w:sz in m:rPr
                    w_sz = rPr.find(qn("w:sz"))
                    if w_sz is not None:
                        rPr.remove(w_sz)
                    w_szCs = rPr.find(qn("w:szCs"))
                    if w_szCs is not None:
                        rPr.remove(w_szCs)
            para_count += 1
    print(f"  Formatted {para_count} paragraphs.")

    # ── Step 2: Find and replace tables ──
    table_indices = []
    for i, child in enumerate(body):
        if child.tag == qn("w:tbl"):
            table_indices.append(i)
    print(f"  Found {len(table_indices)} tables at body indices: {table_indices}")

    # Build new table XML
    new_tbl1 = make_table1()
    new_tbl2 = make_table2()

    # Replace Table 1 (first table) and Table 2 (second table)
    if len(table_indices) >= 1:
        old_t1 = body[table_indices[0]]
        body.replace(old_t1, new_tbl1)
        print("  Replaced Table 1 (Simulation scenarios)")

    if len(table_indices) >= 2:
        # Index might have shifted if we replaced at an earlier index
        new_indices = [i for i, child in enumerate(body) if child.tag == qn("w:tbl")]
        if len(new_indices) >= 2:
            old_t2 = body[new_indices[1]]
            body.replace(old_t2, new_tbl2)
            print("  Replaced Table 2 (Test MSE)")

    # Apply three-line to remaining tables (3, 4, 5)
    remaining_tables = [child for child in body if child.tag == qn("w:tbl")]
    for ti, tbl in enumerate(remaining_tables[2:], start=3):
        apply_three_line_borders(tbl)
        # Also fix font in table cells
        for tc in tbl.iter(qn("w:tc")):
            set_cell_padding(tc)
        for r in tbl.iter(qn("w:r")):
            set_run_font(r)
        print(f"  Formatted Table {ti} (three-line + font)")

    # ── Step 3: Set margins in sectPr ──
    sectPr = body.find(qn("w:sectPr"))
    if sectPr is None:
        sectPr = etree.SubElement(body, qn("w:sectPr"))
    pgMar = sectPr.find(qn("w:pgMar"))
    if pgMar is None:
        pgMar = etree.SubElement(sectPr, qn("w:pgMar"))
    for side in ["top", "bottom", "left", "right"]:
        pgMar.set(qn(f"w:{side}"), "1440")  # 1 inch

    # Write modified XML
    modified_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    all_files["word/document.xml"] = modified_xml

    # ── Step 4: Write new zip ──
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as z_out:
        for name, data in all_files.items():
            z_out.writestr(name, data)

    # ── Step 5: Verify ──
    print(f"\nVerifying {OUTPUT}...")
    verify_doc = etree.fromstring(all_files["word/document.xml"])
    verify_body = verify_doc.find(qn("w:body"))
    verify_paras = [c for c in verify_body if c.tag == qn("w:p")]
    verify_tables = [c for c in verify_body if c.tag == qn("w:tbl")]
    verify_last = verify_body[-1]
    print(f"  Body children: {len(verify_body)}")
    print(f"  Paragraphs: {len(verify_paras)}")
    print(f"  Tables: {len(verify_tables)}")
    print(f"  Last child tag: {verify_last.tag.split('}')[1]}")
    print(f"  File size: {os.path.getsize(OUTPUT)} bytes")

    # Word count
    word_count = 0
    for p_tag in verify_paras:
        texts = [t.text or '' for t in p_tag.iter(qn("w:t"))]
        full_text = ''.join(texts)
        word_count += len(full_text.split())
    print(f"  Word count: ~{word_count}")

    # OMML count
    omath_count = len(list(verify_doc.iter(qn("m:oMath"))))
    print(f"  OMML formulas: {omath_count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
