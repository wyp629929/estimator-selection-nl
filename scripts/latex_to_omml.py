"""
Convert LaTeX math expressions to Word OMML (Office Math Markup Language)
for embedding in python-docx documents.

Pipeline: LaTeX → latex2mathml → MathML → OMML → Word docx element
"""

import re
import latex2mathml.converter
from lxml import etree

OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Function names that should be upright (not italic) in equations
FUNCTION_NAMES = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "lg", "exp",
    "max", "min", "sup", "inf", "lim",
    "det", "rank", "tr", "dim", "ker", "hom",
    "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan",
    "arg", "deg", "mod",
}


def latex_to_omml(latex_str):
    """Convert a LaTeX math string (without $ delimiters) to an OMML oMath element."""
    # Convert LaTeX → MathML
    mathml_str = latex2mathml.converter.convert(latex_str)
    mathml_root = etree.fromstring(mathml_str.encode('utf-8'))

    # Build OMML structure
    omath = etree.Element(f"{{{OMML_NS}}}oMath")
    _convert_children(mathml_root, omath)
    return omath


def _convert_children(mathml_parent, omml_parent):
    """Convert all children of a MathML node, appending OMML nodes to omml_parent."""
    for child in mathml_parent.iterchildren():
        _convert_node(child, omml_parent)


def _convert_node(mathml_node, omml_parent):
    """Convert a single MathML node and append(s) to omml_parent."""
    # Get local tag name (strip namespace)
    tag = mathml_node.tag
    if tag.startswith("{"):
        local_tag = tag.split("}", 1)[1]
    else:
        local_tag = tag

    if local_tag == "mrow":
        # Unwrap — convert children directly into parent
        _convert_children(mathml_node, omml_parent)

    elif local_tag == "mi":
        text = mathml_node.text or ""
        is_func = text.strip() in FUNCTION_NAMES
        r = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        if is_func:
            rpr = etree.SubElement(r, f"{{{OMML_NS}}}rPr")
            sty = etree.SubElement(rpr, f"{{{OMML_NS}}}sty")
            sty.set(f"{{{OMML_NS}}}val", "p")  # plain (upright)
        t = etree.SubElement(r, f"{{{OMML_NS}}}t")
        _set_text_preserve(t, text)

    elif local_tag == "mn":
        r = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        rpr = etree.SubElement(r, f"{{{OMML_NS}}}rPr")
        sty = etree.SubElement(rpr, f"{{{OMML_NS}}}sty")
        sty.set(f"{{{OMML_NS}}}val", "p")  # plain (upright) for numbers
        t = etree.SubElement(r, f"{{{OMML_NS}}}t")
        _set_text_preserve(t, mathml_node.text or "")

    elif local_tag == "mo":
        text = mathml_node.text or ""
        r = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        t = etree.SubElement(r, f"{{{OMML_NS}}}t")
        _set_text_preserve(t, text)

    elif local_tag == "mtext":
        r = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        t = etree.SubElement(r, f"{{{OMML_NS}}}t")
        _set_text_preserve(t, mathml_node.text or "")

    elif local_tag == "msub":
        elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}sSub")
        # Base
        e_base = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[0], e_base)
        # Subscript
        e_sub = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[1], e_sub)

    elif local_tag == "msup":
        elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}sSup")
        e_base = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[0], e_base)
        e_sup = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[1], e_sup)

    elif local_tag == "msubsup":
        elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}sSubSup")
        e_base = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[0], e_base)
        e_sub = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[1], e_sub)
        e_sup = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_node(mathml_node[2], e_sup)

    elif local_tag == "mfrac":
        elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}f")
        num = etree.SubElement(elem, f"{{{OMML_NS}}}num")
        _convert_node(mathml_node[0], num)
        den = etree.SubElement(elem, f"{{{OMML_NS}}}den")
        _convert_node(mathml_node[1], den)

    elif local_tag == "msqrt":
        elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}rad")
        # OMML rad requires deg (empty for square root) + e
        deg = etree.SubElement(elem, f"{{{OMML_NS}}}deg")
        e = etree.SubElement(elem, f"{{{OMML_NS}}}e")
        _convert_children(mathml_node, e)

    elif local_tag == "mstyle":
        _convert_children(mathml_node, omml_parent)

    elif local_tag == "mfenced":
        # Convert to delimiter with fence characters
        # Get the open/close from attributes
        open_ch = mathml_node.get("open", "(")
        close_ch = mathml_node.get("close", ")")

        r = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        t = etree.SubElement(r, f"{{{OMML_NS}}}t")
        _set_text_preserve(t, open_ch)

        for child in mathml_node:
            _convert_node(child, omml_parent)

        r2 = etree.SubElement(omml_parent, f"{{{OMML_NS}}}r")
        t2 = etree.SubElement(r2, f"{{{OMML_NS}}}t")
        _set_text_preserve(t2, close_ch)

    elif local_tag == "munderover":
        # Convert to OMML's limLow/limUpp or bar
        # For simplicity, handle basic cases
        if len(mathml_node) >= 3:
            elem = etree.SubElement(omml_parent, f"{{{OMML_NS}}}limUpp")
            e_base = etree.SubElement(elem, f"{{{OMML_NS}}}e")
            _convert_node(mathml_node[0], e_base)
            e_upper = etree.SubElement(elem, f"{{{OMML_NS}}}lim")
            _convert_node(mathml_node[2], e_upper)
            if len(mathml_node) > 3:
                e_lower = etree.SubElement(elem, f"{{{OMML_NS}}}lim")
                _convert_node(mathml_node[1], e_lower)

    else:
        # Fallback: try to convert children, skip unrecognized elements
        _convert_children(mathml_node, omml_parent)


def _set_text_preserve(t_element, text):
    """Set text and add xml:space='preserve' attribute."""
    t_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_element.text = text


def parse_latex_segments(cell_text):
    """
    Parse text containing $...$ LaTeX segments.
    Returns list of (text, is_math) tuples.
    """
    segments = []
    pattern = re.compile(r'\$(.*?)\$')
    last_end = 0
    for match in pattern.finditer(cell_text):
        # Text before this formula
        if match.start() > last_end:
            segments.append((cell_text[last_end:match.start()], False))
        # The formula (without $ delimiters)
        segments.append((match.group(1), True))
        last_end = match.end()
    # Remaining text after last formula
    if last_end < len(cell_text):
        segments.append((cell_text[last_end:], False))
    return segments


def build_cell_paragraph(cell_text, bold=False, align_val="center"):
    """
    Build a Word paragraph (w:p) element with mixed text and OMML formula objects.
    Handles $...$ segments as inline equations.
    """
    p = etree.Element(f"{{{W_NS}}}p")

    # Paragraph properties (alignment)
    pPr = etree.SubElement(p, f"{{{W_NS}}}pPr")
    jc = etree.SubElement(pPr, f"{{{W_NS}}}jc")
    jc.set(f"{{{W_NS}}}val", align_val)

    segments = parse_latex_segments(cell_text)
    if not segments:
        return p

    for text, is_math in segments:
        if not text:
            continue
        if is_math:
            try:
                omath = latex_to_omml(text)
                # Wrap in oMathPara for inline display
                omath_para = etree.SubElement(p, f"{{{OMML_NS}}}oMathPara")
                omath_para.append(omath)
            except Exception as e:
                # Fallback: show LaTeX source as text
                r = etree.SubElement(p, f"{{{W_NS}}}r")
                rPr = etree.SubElement(r, f"{{{W_NS}}}rPr")
                rFonts = etree.SubElement(rPr, f"{{{W_NS}}}rFonts")
                rFonts.set(f"{{{W_NS}}}ascii", "Times New Roman")
                rFonts.set(f"{{{W_NS}}}hAnsi", "Times New Roman")
                sz = etree.SubElement(rPr, f"{{{W_NS}}}sz")
                sz.set(f"{{{W_NS}}}val", "21")
                if bold:
                    etree.SubElement(rPr, f"{{{W_NS}}}b")
                t = etree.SubElement(r, f"{{{W_NS}}}t")
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                t.text = f"${text}$"
        else:
            # Regular text
            r = etree.SubElement(p, f"{{{W_NS}}}r")
            rPr = etree.SubElement(r, f"{{{W_NS}}}rPr")
            rFonts = etree.SubElement(rPr, f"{{{W_NS}}}rFonts")
            rFonts.set(f"{{{W_NS}}}ascii", "Times New Roman")
            rFonts.set(f"{{{W_NS}}}hAnsi", "Times New Roman")
            sz = etree.SubElement(rPr, f"{{{W_NS}}}sz")
            sz.set(f"{{{W_NS}}}val", "21")
            if bold:
                etree.SubElement(rPr, f"{{{W_NS}}}b")
            t = etree.SubElement(r, f"{{{W_NS}}}t")
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            t.text = text

    return p


def set_cell_with_math(tc, text, bold=False, align_val="center"):
    """Set a table cell's content with math support."""
    # Remove existing paragraphs
    for p in tc.findall(f"{{{W_NS}}}p"):
        tc.remove(p)
    # Build and append new paragraph
    new_p = build_cell_paragraph(text, bold=bold, align_val=align_val)
    tc.append(new_p)
