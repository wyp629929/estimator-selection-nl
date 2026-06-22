"""
Reformat ALL tables in main.docx as academic three-line tables (三线表).

For each table:
  - Remove all vertical borders (left, right, insideV)
  - Remove internal horizontal borders (insideH)
  - Set top border: thick 1.5pt (sz=12)
  - Set bottom border: thick 1.5pt (sz=12)
  - Add header-bottom border on first row: thin 0.75pt (sz=6)
  - Bold + center-align header row
  - Set cell padding
  - For Table 2 (MSE): fix scenario names, merge scenario column
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import parse_xml, OxmlElement

from latex_to_omml import set_cell_with_math

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(10)

# ── Correct MSE data for Table 2 ──────────────────────────────────────────
MSE_DATA = [
    # (scenario, method, mse, sd)
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

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {"w": W_NS}


def set_three_line_borders(table):
    """Apply three-line table borders to a table."""
    tbl = table._tbl
    tbl_pr = tbl.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = parse_xml(f'<w:tblPr {nsdecls("w")}></w:tblPr>')
        tbl.insert(0, tbl_pr)

    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '</w:tblBorders>'
    )
    existing = tbl_pr.find(qn('w:tblBorders'))
    if existing is not None:
        tbl_pr.remove(existing)
    tbl_pr.append(borders)


def set_cell_padding_xml(tc, top=2, bottom=2, left=4, right=4):
    """Set cell padding on a tc element."""
    tc_pr = tc.find(qn('w:tcPr'))
    if tc_pr is None:
        tc_pr = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc.insert(0, tc_pr)
    margins = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top * 20}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom * 20}" w:type="dxa"/>'
        f'  <w:left w:w="{left * 20}" w:type="dxa"/>'
        f'  <w:right w:w="{right * 20}" w:type="dxa"/>'
        '</w:tcMar>'
    )
    existing_mar = tc_pr.find(qn('w:tcMar'))
    if existing_mar is not None:
        tc_pr.remove(existing_mar)
    tc_pr.append(margins)


def set_cell_border_bottom(tc, sz="6"):
    """Add thin bottom border to a tc element."""
    tc_pr = tc.find(qn('w:tcPr'))
    if tc_pr is None:
        tc_pr = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc.insert(0, tc_pr)
    cell_borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        '</w:tcBorders>'
    )
    existing_borders = tc_pr.find(qn('w:tcBorders'))
    if existing_borders is not None:
        tc_pr.remove(existing_borders)
    tc_pr.append(cell_borders)


def clear_cell_text(tc):
    """Remove all paragraphs from a tc element."""
    for p in tc.findall(qn('w:p')):
        tc.remove(p)
    # Add a single empty paragraph with empty run
    p = OxmlElement('w:p')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    p.append(r)
    tc.append(p)


def set_cell_text(tc, text, bold=False, align_val="center", font_name=FONT_NAME, font_size=21):
    """Set cell text with formatting. font_size in half-points (21 = 10.5pt)."""
    # Also set alignment via w:jc in pPr
    # Remove existing paragraphs
    for p in tc.findall(qn('w:p')):
        tc.remove(p)

    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')

    # Alignment
    jc = OxmlElement('w:jc')
    jc.set(qn('w:val'), align_val)
    pPr.append(jc)

    p.append(pPr)

    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    # Font name
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rPr.append(rFonts)
    # Font size
    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), str(font_size))
    rPr.append(sz)
    # Bold
    if bold:
        b = OxmlElement('w:b')
        rPr.append(b)
    r.append(rPr)

    t = OxmlElement('w:t')
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    r.append(t)
    p.append(r)
    tc.append(p)


def get_tc_at(tbl, row_idx, col_idx):
    """Get the actual tc element at (row_idx, col_idx) using lxml."""
    trs = tbl.findall(qn('w:tr'))
    if row_idx >= len(trs):
        return None
    tr = trs[row_idx]
    # Get all tc children of this tr (including ones with gridSpan)
    tcs = tr.findall(qn('w:tc'))
    if col_idx < len(tcs):
        return tcs[col_idx]
    return None


def nsdecls(prefix):
    return f'xmlns:{prefix}="{W_NS}"'


# ── Main ──────────────────────────────────────────────────────────────────
INPUT = "/Users/wangyaoping/ml-inference-paper/paper/main.docx"
OUTPUT = "/Users/wangyaoping/ml-inference-paper/paper/main.docx"

doc = Document(INPUT)

for ti, table in enumerate(doc.tables):
    tbl = table._tbl
    trs = tbl.findall(qn('w:tr'))

    print(f"===== Table {ti+1}: {len(trs)} rows =====")

    # ── Step A: Three-line table borders ─────────────────────────────
    set_three_line_borders(table)

    # ── Step B: Cell padding + bottom border on header row ───────────
    for ri, tr in enumerate(trs):
        tcs = tr.findall(qn('w:tc'))
        for ci, tc in enumerate(tcs):
            set_cell_padding_xml(tc)
            if ri == 0:
                set_cell_border_bottom(tc)

    # ── Step C: Table-specific content formatting ────────────────────
    if ti == 0:  # Table 1: Simulation scenarios (5 rows)
        # Header
        header_texts = ["Scenario", "Data generating process", "Purpose"]
        tcs = trs[0].findall(qn('w:tc'))
        for ci, text in enumerate(header_texts):
            if ci < len(tcs):
                set_cell_text(tcs[ci], text, bold=True, align_val="center")

        # Data rows — column 1 (DGP) contains $...$ LaTeX formulas
        data = [
            ("Linear",      "$y = X\\beta + \\varepsilon$, $\\beta = (2, -1.5, 0.8, 0, 0)$", "Baseline: OLS optimal"),
            ("Semiparametric", "$y = X\\beta + 0.3\\sin(X_1) + \\varepsilon$", "Mild misspecification"),
            ("Nonlinear",   "$y = \\sin X_1 + \\log(1+|X_2|) + X_3 X_4 + \\varepsilon$", "Strong nonlinearity"),
            ("High-dim sparse", "$p=100$, 5 non-zero coefficients", "Selection + estimation"),
        ]
        for ri, (scenario, dgp, purpose) in enumerate(data, start=1):
            if ri < len(trs):
                tcs = trs[ri].findall(qn('w:tc'))
                if len(tcs) >= 3:
                    set_cell_text(tcs[0], scenario, align_val="left")
                    set_cell_with_math(tcs[1], dgp, align_val="left")  # handles $...$ → OMML
                    set_cell_text(tcs[2], purpose, align_val="left")

    elif ti == 1:  # Table 2: Test MSE (may need cleanup if still 35 rows)
        # Check if cleanup is needed (original had 35 rows, clean is 33)
        if len(trs) == 35:
            # Remove extra rows: row 34 (empty), then row 1 (sub-header)
            extra_last = trs[34]
            tbl.remove(extra_last)
            trs = tbl.findall(qn('w:tr'))  # 34 rows now
            sub_header = trs[1]
            tbl.remove(sub_header)
            trs = tbl.findall(qn('w:tr'))  # 33 rows now
            print(f"   Cleaned up: {len(trs)} rows")
        elif len(trs) == 33:
            print(f"   Already clean: {len(trs)} rows")

        print(f"   After cleanup: {len(trs)} rows")

        # Header row (index 0)
        tcs = trs[0].findall(qn('w:tc'))
        for ci, text in enumerate(["scenario", "method", "MSE", "SD"]):
            if ci < len(tcs):
                set_cell_text(tcs[ci], text, bold=True, align_val="center")

        # Data rows (indices 1-32) with correct scenario names
        for ri, (scenario, method, mse, sd) in enumerate(MSE_DATA, start=1):
            if ri < len(trs):
                tcs = trs[ri].findall(qn('w:tc'))
                if len(tcs) >= 4:
                    set_cell_text(tcs[0], scenario, align_val="left")
                    set_cell_text(tcs[1], method, align_val="left")
                    set_cell_text(tcs[2], mse, align_val="center")
                    set_cell_text(tcs[3], sd, align_val="center")

        # Vertical merge scenario column (4 groups of 8 rows)
        trs = tbl.findall(qn('w:tr'))  # refresh
        for start_r in [1, 9, 17, 25]:
            for r in range(start_r, start_r + 8):
                tc = trs[r].findall(qn('w:tc'))[0]
                tc_pr = tc.find(qn('w:tcPr'))
                if tc_pr is None:
                    tc_pr = OxmlElement('w:tcPr')
                    tc.insert(0, tc_pr)

                if r == start_r:
                    vm = OxmlElement('w:vMerge')
                    vm.set(qn('w:val'), 'restart')
                    va = OxmlElement('w:vAlign')
                    va.set(qn('w:val'), 'center')
                    existing_vm = tc_pr.find(qn('w:vMerge'))
                    if existing_vm is not None:
                        tc_pr.remove(existing_vm)
                    existing_va = tc_pr.find(qn('w:vAlign'))
                    if existing_va is not None:
                        tc_pr.remove(existing_va)
                    tc_pr.append(vm)
                    tc_pr.append(va)
                else:
                    vm = OxmlElement('w:vMerge')
                    vm.set(qn('w:val'), 'continue')
                    existing_vm = tc_pr.find(qn('w:vMerge'))
                    if existing_vm is not None:
                        tc_pr.remove(existing_vm)
                    tc_pr.append(vm)
                    clear_cell_text(tc)

        # Set column widths
        widths = [Inches(1.3), Inches(1.1), Inches(1.0), Inches(1.0)]
        for row in table.rows:
            for ci, w in enumerate(widths):
                row.cells[ci].width = w

    elif ti == 2:  # Table 3: Home Credit (9 rows)
        tcs = trs[0].findall(qn('w:tc'))
        for ci, text in enumerate(["Method", "Test MSE", "Train Time (s)"]):
            if ci < len(tcs):
                set_cell_text(tcs[ci], text, bold=True, align_val="center")
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn('w:tc'))
            if len(tcs) >= 3:
                t0 = tcs[0].find('.//' + qn('w:t'))
                t1 = tcs[1].find('.//' + qn('w:t'))
                t2 = tcs[2].find('.//' + qn('w:t'))
                set_cell_text(tcs[0], t0.text.strip() if t0 is not None else '', align_val="left")
                set_cell_text(tcs[1], t1.text.strip() if t1 is not None else '', align_val="center")
                set_cell_text(tcs[2], t2.text.strip() if t2 is not None else '', align_val="center")

    elif ti == 3:  # Table 4: PIMA Diabetes (10 rows → 9)
        if len(trs) == 10:
            sub_header = trs[1]
            tbl.remove(sub_header)
            trs = tbl.findall(qn('w:tr'))
            print(f"   Cleaned up: {len(trs)} rows")
        else:
            print(f"   Already clean: {len(trs)} rows")

        tcs = trs[0].findall(qn('w:tc'))
        for ci, text in enumerate(["Method", "Test MSE"]):
            if ci < len(tcs):
                set_cell_text(tcs[ci], text, bold=True, align_val="center")
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn('w:tc'))
            if len(tcs) >= 2:
                t0 = tcs[0].find('.//' + qn('w:t'))
                t1 = tcs[1].find('.//' + qn('w:t'))
                set_cell_text(tcs[0], t0.text.strip() if t0 is not None else '', align_val="left")
                set_cell_text(tcs[1], t1.text.strip() if t1 is not None else '', align_val="center")

    elif ti == 4:  # Table 5: Computation cost (9 rows)
        tcs = trs[0].findall(qn('w:tc'))
        for ci, text in enumerate(["Method", "Train Time (s)"]):
            if ci < len(tcs):
                set_cell_text(tcs[ci], text, bold=True, align_val="center")
        for ri in range(1, len(trs)):
            tcs = trs[ri].findall(qn('w:tc'))
            if len(tcs) >= 2:
                t0 = tcs[0].find('.//' + qn('w:t'))
                t1 = tcs[1].find('.//' + qn('w:t'))
                set_cell_text(tcs[0], t0.text.strip() if t0 is not None else '', align_val="left")
                set_cell_text(tcs[1], t1.text.strip() if t1 is not None else '', align_val="center")

    print(f"   Done — {len(trs)} rows × {len(trs[0].findall(qn('w:tc')))} cols")

# ── Save ──────────────────────────────────────────────────────────────────
doc.save(OUTPUT)
print(f"\nSaved: {OUTPUT}")
