"""Convert documentation/RESEARCH.md to a Word .docx using only the stdlib.

A .docx is a ZIP of XML parts, so no third-party dependency is needed -- which
matters here because this project runs on Python 3.14, where python-docx's lxml
dependency has to build from source.

Handles the Markdown subset RESEARCH.md actually uses: ATX headings, pipe
tables, blockquotes, bullet/ordered lists, fenced code, horizontal rules, and
inline bold/italic/code/links. Not a general Markdown engine -- extend it
alongside the document.

Usage:
    python training/md_to_docx.py                 # RESEARCH.md -> RESEARCH.docx
    python training/md_to_docx.py in.md out.docx
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# Inline: `code`, **bold**, *italic*, [text](url). Code first, so emphasis
# markers inside backticks survive untouched.
INLINE = re.compile(r"(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RULE = re.compile(r"-{3,}|\*{3,}|_{3,}")
BLOCK_START = re.compile(r"(#{1,6})\s|[-*+]\s|\d+[.)]\s|\||>|```")

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


class Docx:
    """Accumulates body XML and the external-link relationships it references."""

    def __init__(self) -> None:
        self.body: list[str] = []
        self.rels: list[tuple[str, str]] = []  # (rId, url)

    def rel_for(self, url: str) -> str:
        rid = f"rId{100 + len(self.rels)}"
        self.rels.append((rid, url))
        return rid

    # -- inline ---------------------------------------------------------
    def runs(self, text: str, *, bold=False, italic=False, size=None) -> str:
        out = []
        for part in (p for p in INLINE.split(text) if p):
            if len(part) > 1 and part.startswith("`") and part.endswith("`"):
                out.append(self._run(part[1:-1], mono=True, size=size or 19))
            elif part.startswith("**") and part.endswith("**"):
                out.append(self._run(part[2:-2], bold=True, italic=italic, size=size))
            elif len(part) > 2 and part.startswith("*") and part.endswith("*"):
                out.append(self._run(part[1:-1], italic=True, bold=bold, size=size))
            elif m := LINK.fullmatch(part):
                rid = self.rel_for(m.group(2))
                inner = self._run(m.group(1), color="0B5CAB", underline=True, size=size)
                out.append(f'<w:hyperlink r:id="{rid}">{inner}</w:hyperlink>')
            else:
                out.append(self._run(part, bold=bold, italic=italic, size=size))
        return "".join(out)

    @staticmethod
    def _run(
        text: str, *, bold=False, italic=False, mono=False,
        color=None, underline=False, size=None,
    ) -> str:
        props = []
        if mono:
            props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
        if bold:
            props.append("<w:b/>")
        if italic:
            props.append("<w:i/>")
        if underline:
            props.append('<w:u w:val="single"/>')
        if color:
            props.append(f'<w:color w:val="{color}"/>')
        if mono and not color:
            props.append('<w:color w:val="333333"/>')
        if size:
            props.append(f'<w:sz w:val="{size}"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        # xml:space preserves the leading/trailing spaces between inline spans.
        return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

    # -- blocks ---------------------------------------------------------
    def para(self, text="", *, style=None, align=None, space_after=120, **kw) -> None:
        props = []
        if style:
            props.append(f'<w:pStyle w:val="{style}"/>')
        if align:
            props.append(f'<w:jc w:val="{align}"/>')
        props.append(f'<w:spacing w:after="{space_after}"/>')
        ppr = f"<w:pPr>{''.join(props)}</w:pPr>"
        self.body.append(f"<w:p>{ppr}{self.runs(text, **kw) if text else ''}</w:p>")

    def heading(self, text: str, level: int) -> None:
        self.para(text, style=f"Heading{min(level, 4)}",
                  align="center" if level == 1 else None)

    def listitem(self, text: str, ordered: bool) -> None:
        # numId 1/2 are defined in numbering.xml below.
        ppr = (
            '<w:pPr><w:pStyle w:val="ListParagraph"/>'
            f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{2 if ordered else 1}"/></w:numPr>'
            "<w:spacing w:after=\"60\"/></w:pPr>"
        )
        self.body.append(f"<w:p>{ppr}{self.runs(text)}</w:p>")

    def quote(self, text: str) -> None:
        ppr = (
            '<w:pPr><w:ind w:left="480"/><w:spacing w:after="160"/>'
            '<w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="4472C4"/></w:pBdr>'
            "</w:pPr>"
        )
        self.body.append(f"<w:p>{ppr}{self.runs(text, italic=True)}</w:p>")

    def code(self, lines: list[str]) -> None:
        ppr = (
            '<w:pPr><w:shd w:val="clear" w:fill="F5F5F5"/>'
            '<w:spacing w:after="120"/></w:pPr>'
        )
        runs = "<w:r><w:br/></w:r>".join(
            self._run(ln, mono=True, size=17) for ln in lines
        )
        self.body.append(f"<w:p>{ppr}{runs}</w:p>")

    def table(self, rows: list[list[str]]) -> None:
        width = max(len(r) for r in rows)
        borders = "".join(
            f'<w:{e} w:val="single" w:sz="4" w:color="BFBFBF"/>'
            for e in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        xml = [
            "<w:tbl><w:tblPr>"
            '<w:tblW w:w="5000" w:type="pct"/>'
            f"<w:tblBorders>{borders}</w:tblBorders>"
            "</w:tblPr>"
        ]
        for i, row in enumerate(rows):
            xml.append("<w:tr>")
            for j in range(width):
                cell = row[j] if j < len(row) else ""
                shade = '<w:shd w:val="clear" w:fill="EDEDED"/>' if i == 0 else ""
                xml.append(
                    f"<w:tc><w:tcPr>{shade}</w:tcPr>"
                    f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>'
                    f"{self.runs(cell, bold=(i == 0), size=18)}</w:p></w:tc>"
                )
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))
        self.para(space_after=0)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_divider(line: str) -> bool:
    cells = split_row(line)
    return bool(cells) and all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def parse(md: str, d: Docx) -> None:
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if not s:
            i += 1
            continue

        if RULE.fullmatch(s):
            d.para(space_after=200)
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            d.code(buf)
            continue

        if s.startswith("|") and i + 1 < len(lines) and is_divider(lines[i + 1]):
            rows = [split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            d.table(rows)
            continue

        if m := re.match(r"(#{1,6})\s+(.*)", s):
            d.heading(m.group(2), len(m.group(1)))
            i += 1
            continue

        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            d.quote(" ".join(buf))
            continue

        if m := re.match(r"\d+[.)]\s+(.*)", s):
            d.listitem(m.group(1), ordered=True)
            i += 1
            continue

        if m := re.match(r"[-*+]\s+(.*)", s):
            d.listitem(m.group(1), ordered=False)
            i += 1
            continue

        # Paragraph: join soft-wrapped lines until a blank or new block.
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or BLOCK_START.match(nxt) or RULE.fullmatch(nxt):
                break
            buf.append(nxt)
            i += 1
        d.para(" ".join(buf))


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def styles_xml() -> str:
    heads = []
    for lvl, (sz, color, before) in enumerate(
        [(36, "1F3864", 240), (30, "2E5496", 240), (25, "2E5496", 200), (23, "404040", 180)],
        start=1,
    ):
        heads.append(
            f'<w:style w:type="paragraph" w:styleId="Heading{lvl}">'
            f'<w:name w:val="heading {lvl}"/><w:basedOn w:val="Normal"/>'
            f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{lvl - 1}"/>'
            f'<w:spacing w:before="{before}" w:after="120"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>'
            f'<w:b/><w:sz w:val="{sz}"/><w:color w:val="{color}"/></w:rPr></w:style>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {W}>
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/>
</w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
<w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>
<w:basedOn w:val="Normal"/></w:style>
{''.join(heads)}
</w:styles>"""


NUMBERING = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering {W}>
<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0">
<w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="&#8226;"/>
<w:pPr><w:ind w:left="480" w:hanging="240"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr></w:lvl></w:abstractNum>
<w:abstractNum w:abstractNumId="2"><w:lvl w:ilvl="0">
<w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>
<w:pPr><w:ind w:left="480" w:hanging="240"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>
<w:num w:numId="2"><w:abstractNumId w:val="2"/></w:num>
</w:numbering>"""


def build(md_path: Path, docx_path: Path) -> None:
    d = Docx()
    parse(md_path.read_text(encoding="utf-8"), d)

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {W} {R}><w:body>{''.join(d.body)}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
        "</w:sectPr></w:body></w:document>"
    )

    links = "".join(
        f'<Relationship Id="{rid}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        f'Target="{escape(url, {chr(34): "&quot;"})}" TargetMode="External"/>'
        for rid, url in d.rels
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
        f"{links}</Relationships>"
    )

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", NUMBERING)
        z.writestr("word/_rels/document.xml.rels", doc_rels)

    kb = docx_path.stat().st_size / 1024
    print(f"Wrote {docx_path} ({kb:.1f} KB, {len(d.rels)} hyperlinks)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "documentation" / "RESEARCH.md"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".docx")
    build(src, dst)
