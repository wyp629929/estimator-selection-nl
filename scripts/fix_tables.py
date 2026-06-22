"""
Precision fix script for main.docx — ONLY modifies table cells,
preserves ALL body text, references, and formatting exactly as-is.

Dependencies:
  pip install latex2mathml

What this does:
  1. Applies three-line table borders to all 5 tables
  2. Converts $...$ LaTeX in Table 1 (DGP column) to Word OMML formulas
  3. Merges identical scenario cells vertically in Table 2
  4. Saves without touching any body paragraph
"""

import re, os, copy
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import parse_xml, OxmlElement

import latex2mathml.converter
from lxml import etree

# ── Namespaces ────────────────────────────────────────────────────────────
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# Known function names → upright (not italic) in equations
FUNCTION_NAMES = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "lg", "exp",
    "max", "min", "sup", "inf", "lim",
    "det", "rank", "tr", "dim",
    "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan",
}

# ── Correct MSE data for Table 2 ──────────────────────────────────────────
MSE_DATA = [
    ("highdim", "dnn", "3.3050", "0.6942"),
    ("highdim", "lasso", "1.0735", "0.1510"),
    ("highdim", "lightgbm", "3.5685", "0.8672"),
    ("highdim", "mlp", "7.9618", "1.8603"),
    ("highdim", "ols", "1.3985", "0.1913"),
    ("highdim", "rf", "7.1531", "2.0531"),
    ("highdim", "ridge", "1.3938", "0.1902"),
    ("highdim", "xgboost", "6.3345", "1.7439"),
    ("linear", "dnn", "1.2156", "0.1572"),
    ("linear", "lasso", "1.0516", "0.1234"),
    ("linear", "lightgbm", "1.6059", "0.2051"),
    ("linear", "mlp", "1.1082", "0.1286"),
    ("linear", "ols", "1.0336", "0.1224"),
    ("linear", "rf", "1.5009", "0.2001"),
    ("linear", "ridge", "1.0333", "0.1224"),
    ("linear", "xgboost", "1.6309", "0.2179"),
    ("nonlinear", "dnn", "1.3440", "0.1551"),
    ("nonlinear", "lasso", "2.1548", "0.3204"),
    ("nonlinear", "lightgbm", "1.5530", "0.2196"),
    ("nonlinear", "mlp", "1.2196", "0.1528"),
    ("nonlinear", "ols", "2.1912", "0.3295"),
    ("nonlinear", "rf", "1.7650", "0.3085"),
    ("nonlinear", "ridge", "2.1908", "0.3293"),
    ("nonlinear", "xgboost", "1.7451", "0.2731"),
    ("semiparametric", "dnn", "1.2080", "0.1382"),
    ("semiparametric", "lasso", "1.0555", "0.1270"),
    ("semiparametric", "lightgbm", "1.6102", "0.1947"),
    ("semiparametric", "mlp", "1.0853", "0.1223"),
    ("semiparametric", "ols", "1.0232", "0.1142"),
    ("semiparametric", "rf", "1.5374", "0.2077"),
    ("semiparametric", "ridge", "1.0234", "0.1145"),
    ("semiparametric", "xgboost", "1.6723", "0.2090"),
]

# ── Table 1 DGP data (with LaTeX formulas) ────────────────────────────────
TABLE1_DGP = [
    "$y = X\\beta + \\varepsilon$, $\\beta = (2, -1.5, 0.8, 0, 0)$",
    "$y = X\\beta + 0.3\\sin(X_1) + \\varepsilon$",
    "$y = \\sin X_1 + \\log(1+|X_2|) + X_3 X_4 + \\varepsilon$",
    "$p=100$, 5 non-zero coefficients",
]

# ══════════════════════════════════════════════════════════════════════════
#  OMML CONVERSION
# ══════════════════════════════════════════════════════════════════════════

def latex_to_omml(latex_str):
    """LaTeX string → OMML oMath element (without $ delimiters)."""
    mathml = latex2mathml.converter.convert(latex_str)
    mathml_root = etree.fromstring(mathml.encode("utf-8"))
    omath = etree.Element(f"{{{M_NS}}}oMath")
    _convert_children(mathml_root, omath)
    return omath


def _convert_children(src, dst):
    for child in src.iterchildren():
        _convert_node(child, dst)


def _convert_node(node, parent):
    tag = node.tag.split("}", 1)[1] if "}" in node.tag else node.tag

    if tag == "mrow":
        _convert_children(node, parent)

    elif tag in ("mi", "mn", "mo", "mtext"):
        text = node.text or ""
        is_func = (tag == "mi" and text.strip() in FUNCTION_NAMES)
        is_num = (tag == "mn")
        r = etree.SubElement(parent, f"{{{M_NS}}}r")
        if is_func or is_num:
            rpr = etree.SubElement(r, f"{{{M_NS}}}rPr")
            sty = etree.SubElement(rpr, f"{{{M_NS}}}sty")
            sty.set(f"{{{M_NS}}}val", "p")
        t = etree.SubElement(r, f"{{{M_NS}}}t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text

    elif tag == "msub":
        e = etree.SubElement(parent, f"{{{M_NS}}}sSub")
        e1 = etree.SubElement(e, f"{{{M_NS}}}e")
        _convert_node(node[0], e1)
        e2 = etree.SubElement(e, f"{{{M_NS}}}e")
        _convert_node(node[1], e2)

    elif tag == "msup":
        e = etree.SubElement(parent, f"{{{M_NS}}}sSup")
        e1 = etree.SubElement(e, f"{{{M_NS}}}e")
        _convert_node(node[0], e1)
        e2 = etree.SubElement(e, f"{{{M_NS}}}e")
        _convert_node(node[1], e2)

    elif tag == "msubsup":
        e = etree.SubElement(parent, f"{{{M_NS}}}sSubSup")
        for i in range(3):
            ec = etree.SubElement(e, f"{{{M_NS}}}e")
            _convert_node(node[i], ec)

    elif tag == "mfrac":
        e = etree.SubElement(parent, f"{{{M_NS}}}f")
        num = etree.SubElement(e, f"{{{M_NS}}}num")
        _convert_node(node[0], num)
        den = etree.SubElement(e, f"{{{M_NS}}}den")
        _convert_node(node[1], den)

    elif tag in ("mstyle", "merror", "mphantom"):
        _convert_children(node, parent)

    elif tag == "mfenced":
        open_ch = node.get("open", "(")
        close_ch = node.get("close", ")")
        r1 = etree.SubElement(parent, f"{{{M_NS}}}r")
        t1 = etree.SubElement(r1, f"{{{M_NS}}}t")
        t1.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t1.text = open_ch
        for child in node:
            _convert_node(child, parent)
        r2 = etree.SubElement(parent, f"{{{M_NS}}}r")
        t2 = etree.SubElement(r2, f"{{{M_NS}}}t")
        t2.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t2.text = close_ch
    else:
        _convert_children(node, parent)


def parse_latex_segments(text):
    """Split text into [(content, is_math), ...] segments based on $...$."""
    segments = []
    pattern = re.compile(r"\$(.*?)\$")
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            segments.append((text[last:m.start()], False))
        segments.append((m.group(1), True))
        last = m.end()
    if last < len(text):
        segments.append((text[last:], False))
    return segments


def build_paragraph_with_math(text, align_val="left"):
    """Build a w:p element with mixed OMML + text runs from $...$ segments."""
    p = etree.Element(f"{{{W_NS}}}p")
    pPr = etree.SubElement(p, f"{{{W_NS}}}pPr")
    jc = etree.SubElement(pPr, f"{{{W_NS}}}jc")
    jc.set(f"{{{W_NS}}}val", align_val)

    for content, is_math in parse_latex_segments(text):
        if not content:
            continue
        if is_math:
            try:
                omath = latex_to_omml(content)
                omp = etree.SubElement(p, f"{{{M_NS}}}oMathPara")
                omp.append(omath)
            except Exception:
                # Fallback: show raw LaTeX as text
                r = _make_run(content, italic=False)
                p.append(r)
        else:
            r = _make_run(content, italic=False)
            p.append(r)
    return p


def _make_run(text, bold=False, italic=False):
    """Create a w:r run element."""
    r = etree.Element(f"{{{W_NS}}}r")
    rPr = etree.SubElement(r, f"{{{W_NS}}}rPr")
    rFonts = etree.SubElement(rPr, f"{{{W_NS}}}rFonts")
    rFonts.set(f"{{{W_NS}}}ascii", "Times New Roman")
    rFonts.set(f"{{{W_NS}}}hAnsi", "Times New Roman")
    sz = etree.SubElement(rPr, f"{{{W_NS}}}sz")
    sz.set(f"{{{W_NS}}}val", "20")
    if bold:
        etree.SubElement(rPr, f"{{{W_NS}}}b")
    if italic:
        etree.SubElement(rPr, f"{{{W_NS}}}i")
    t = etree.SubElement(r, f"{{{W_NS}}}t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


# ══════════════════════════════════════════════════════════════════════════
#  THREE-LINE TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def apply_three_line_borders(tbl):
    """Set table-level three-line borders (top/bottom thick, others none)."""
    tbl_pr = tbl.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = parse_xml(f'<w:tblPr {{}}></w:tblPr>'.replace("{}", f'xmlns:w="{W_NS}"'))
        tbl.insert(0, tbl_pr)

    borders = parse_xml(
        f'<w:tblBorders xmlns:w="{W_NS}">'
        '  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        "</w:tblBorders>"
    )
    existing = tbl_pr.find(qn("w:tblBorders"))
    if existing is not None:
        tbl_pr.remove(existing)
    tbl_pr.append(borders)


def set_cell_padding(tc, top=2, bottom=2, left=4, right=4):
    """Set cell margins in points (1pt = 20 twips)."""
    tc_pr = tc.find(qn("w:tcPr"))
    if tc_pr is None:
        tc_pr = parse_xml(f'<w:tcPr xmlns:w="{W_NS}"></w:tcPr>')
        tc.insert(0, tc_pr)
    mar = parse_xml(
        f'<w:tcMar xmlns:w="{W_NS}">'
        f'  <w:top w:w="{top*20}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom*20}" w:type="dxa"/>'
        f'  <w:left w:w="{left*20}" w:type="dxa"/>'
        f'  <w:right w:w="{right*20}" w:type="dxa"/>'
        f"</w:tcMar>"
    )
    existing = tc_pr.find(qn("w:tcMar"))
    if existing is not None:
        tc_pr.remove(existing)
    tc_pr.append(mar)


def set_header_bottom_border(tc):
    """Add 0.75pt bottom border to header cell."""
    tc_pr = tc.find(qn("w:tcPr"))
    if tc_pr is None:
        tc_pr = parse_xml(f'<w:tcPr xmlns:w="{W_NS}"></w:tcPr>')
        tc.insert(0, tc_pr)
    cb = parse_xml(
        f'<w:tcBorders xmlns:w="{W_NS}">'
        '  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        "</w:tcBorders>"
    )
    existing = tc_pr.find(qn("w:tcBorders"))
    if existing is not None:
        tc_pr.remove(existing)
    tc_pr.append(cb)


def set_cell_simple(tc, text, bold=False, align="center"):
    """Replace cell content with plain text (no math)."""
    for p in tc.findall(qn("w:p")):
        tc.remove(p)
    p = etree.Element(f"{{{W_NS}}}p")
    pPr = etree.SubElement(p, f"{{{W_NS}}}pPr")
    jc = etree.SubElement(pPr, f"{{{W_NS}}}jc")
    jc.set(f"{{{W_NS}}}val", align)
    r = _make_run(text, bold=bold)
    p.append(r)
    tc.append(p)


def set_cell_with_math(tc, text, align="left"):
    """Replace cell content with mixed text + OMML formulas."""
    for p in tc.findall(qn("w:p")):
        tc.remove(p)
    new_p = build_paragraph_with_math(text, align_val=align)
    tc.append(new_p)


def vertical_merge(tbl, col, start_row, end_row):
    """
    Vertically merge cells in a column from start_row to end_row (inclusive).
    Uses explicit w:val="continue" to survive python-docx save/load.
    """
    trs = tbl.findall(qn("w:tr"))
    for r in range(start_row, end_row + 1):
        tcs = trs[r].findall(qn("w:tc"))
        if col >= len(tcs):
            continue
        tc = tcs[col]
        tc_pr = tc.find(qn("w:tcPr"))
        if tc_pr is None:
            tc_pr = OxmlElement("w:tcPr")
            tc.insert(0, tc_pr)

        if r == start_row:
            vm = OxmlElement("w:vMerge")
            vm.set(qn("w:val"), "restart")
            va = OxmlElement("w:vAlign")
            va.set(qn("w:val"), "center")
            for tag in ("w:vMerge", "w:vAlign"):
                existing = tc_pr.find(qn(tag))
                if existing is not None:
                    tc_pr.remove(existing)
            tc_pr.append(vm)
            tc_pr.append(va)
        else:
            vm = OxmlElement("w:vMerge")
            vm.set(qn("w:val"), "continue")
            existing = tc_pr.find(qn("w:vMerge"))
            if existing is not None:
                tc_pr.remove(existing)
            tc_pr.append(vm)
            # Clear text in continuation cells
            for p in tc.findall(qn("w:p")):
                tc.remove(p)
            empty = OxmlElement("w:p")
            empty_r = OxmlElement("w:r")
            empty_t = OxmlElement("w:t")
            empty_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            empty_r.append(empty_t)
            empty.append(empty_r)
            tc.append(empty)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

INPUT = "/Users/wangyaoping/ml-inference-paper/paper/main.docx"
OUTPUT = "/Users/wangyaoping/Desktop/main.docx"

doc = Document(INPUT)

# Preserve section properties by reading them before any modification
section = doc.sections[0]
page_w = section.page_width
page_h = section.page_height
top_m = section.top_margin
bot_m = section.bottom_margin
left_m = section.left_margin
right_m = section.right_margin

for ti, table in enumerate(doc.tables):
    tbl = table._tbl
    trs = tbl.findall(qn("w:tr"))
    print(f"Table {ti+1}: {len(trs)} rows")

    # ── Step 1: Three-line borders ────────────────────────────────────
    apply_three_line_borders(tbl)

    # ── Step 2: Cell padding + header bottom border ───────────────────
    for ri, tr in enumerate(trs):
        for tc in tr.findall(qn("w:tc")):
            set_cell_padding(tc)
            if ri == 0:
                set_header_bottom_border(tc)

    # ── Step 3: Table-specific content ─────────────────────────────────
    if ti == 0:  # Table 1: Simulation scenarios
        # Header
        for ci, text in enumerate(["Scenario", "Data generating process", "Purpose"]):
            tcs_head = trs[0].findall(qn("w:tc"))
            if ci < len(tcs_head):
                set_cell_simple(tcs_head[ci], text, bold=True, align="center")

        # Data rows — col 2 (DGP) has LaTeX formulas
        for ri in range(1, 5):
            tcs = trs[ri].findall(qn("w:tc"))
            if len(tcs) >= 3:
                scenarios = ["Linear", "Semiparametric", "Nonlinear", "High-dim sparse"]
                purposes = [
                    "Baseline: OLS optimal",
                    "Mild misspecification",
                    "Strong nonlinearity",
                    "Selection + estimation",
                ]
                if ri <= 4:
                    set_cell_simple(tcs[0], scenarios[ri - 1], align="left")
                    set_cell_with_math(tcs[1], TABLE1_DGP[ri - 1], align="left")
                    set_cell_simple(tcs[2], purposes[ri - 1], align="left")

    elif ti == 1:  # Table 2: Test MSE
        # Header
        tcs_head = trs[0].findall(qn("w:tc"))
        for ci, text in enumerate(["scenario", "method", "MSE", "SD"]):
            if ci < len(tcs_head):
                set_cell_simple(tcs_head[ci], text, bold=True)

        # Data rows
        for ri, (scenario, method, mse, sd) in enumerate(MSE_DATA, start=1):
            if ri < len(trs):
                tcs = trs[ri].findall(qn("w:tc"))
                if len(tcs) >= 4:
                    set_cell_simple(tcs[0], scenario, align="left")
                    set_cell_simple(tcs[1], method, align="left")
                    set_cell_simple(tcs[2], mse)
                    set_cell_simple(tcs[3], sd)

        # Vertical merge: 4 groups of 8 rows
        for start_r in [1, 9, 17, 25]:
            vertical_merge(tbl, 0, start_r, start_r + 7)

    elif ti == 2:  # Table 3: Home Credit
        hd = ["Method", "Test MSE", "Train Time (s)"]
        for ci, text in enumerate(hd):
            tcs = trs[0].findall(qn("w:tc"))
            if ci < len(tcs):
                set_cell_simple(tcs[ci], text, bold=True)
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn("w:tc"))
            if len(tcs) >= 3:
                set_cell_simple(tcs[0], tcs[0].find(f".//{{{W_NS}}}t").text or "", align="left")
                set_cell_simple(tcs[1], tcs[1].find(f".//{{{W_NS}}}t").text or "")
                set_cell_simple(tcs[2], tcs[2].find(f".//{{{W_NS}}}t").text or "")

    elif ti == 3:  # Table 4: PIMA
        for ci, text in enumerate(["Method", "Test MSE"]):
            tcs = trs[0].findall(qn("w:tc"))
            if ci < len(tcs):
                set_cell_simple(tcs[ci], text, bold=True)
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn("w:tc"))
            if len(tcs) >= 2:
                set_cell_simple(tcs[0], tcs[0].find(f".//{{{W_NS}}}t").text or "", align="left")
                set_cell_simple(tcs[1], tcs[1].find(f".//{{{W_NS}}}t").text or "")

    elif ti == 4:  # Table 5: Computation
        for ci, text in enumerate(["Method", "Train Time (s)"]):
            tcs = trs[0].findall(qn("w:tc"))
            if ci < len(tcs):
                set_cell_simple(tcs[ci], text, bold=True)
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn("w:tc"))
            if len(tcs) >= 2:
                set_cell_simple(tcs[0], tcs[0].find(f".//{{{W_NS}}}t").text or "", align="left")
                set_cell_simple(tcs[1], tcs[1].find(f".//{{{W_NS}}}t").text or "")

    print(f"  → Done")

# ── Restore section properties (critical: prevents Word truncation) ──────
section = doc.sections[0]
section.page_width = page_w
section.page_height = page_h
section.top_margin = top_m
section.bottom_margin = bot_m
section.left_margin = left_m
section.right_margin = right_m

# ── Save ──────────────────────────────────────────────────────────────────
doc.save(OUTPUT)
print(f"\nSaved: {OUTPUT}")
print(f"Done — tables formatted with three-line style + OMML formulas.")
