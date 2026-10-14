"""PDF/A-1b helpers for matplotlib figure PDFs (MIT asks for PDF/A-1; ISO 19005-1).

Two things in a matplotlib PDF break PDF/A-1b, and both are fixed here without changing a single rendered pixel:

1. Transparency (ISO 19005-1, 6.4). The thesis style uses no alpha at all, but matplotlib's PDF backend writes an
   ExtGState with stroke alpha ``/CA 0`` for patches drawn with ``edgecolor='none'`` (and fill alpha ``/ca 0`` for
   hollow markers, ``facecolor='none'``). The backend decides whether to stroke or fill from the colour itself and
   never paints those, so the value is never used; ``no_zero_stroke_alpha`` makes the backend write 1 instead.
   Partial alphas (0 < a < 1) are left alone and reported by ``scan_transparency``.
2. Missing ``/CIDSet`` (6.3.5, test 3). With ``pdf.fonttype = 42`` matplotlib embeds each font subset as a Type 0
   font with a CIDFontType2 descendant and no CIDSet. ``add_cidset`` appends an incremental update that adds the
   CIDSet stream built from the font's own CIDToGIDMap. Font programs, glyphs and widths are untouched.

``scan_transparency`` reports any remaining transparency. The LaTeX side (a transparency group that pdfTeX's
graphics driver adds around every included PDF) is handled in the thesis preamble; see ``docs/reproducing.md``.

The CIDSet and scanner code is adapted from the PDF/A-1b checking tools written for the 2026-10-13 thesis build,
which passed veraPDF 1.30.0.
"""
from __future__ import annotations

import collections
import contextlib
import re
import zlib
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def no_zero_stroke_alpha() -> Iterator[None]:
    """While active, matplotlib's PDF backend writes alpha 1 where it would write an unused alpha of exactly 0."""
    from matplotlib.backends import backend_pdf
    orig = backend_pdf.PdfFile.alphaState

    def alphaState(self, alpha):          # noqa: N802 (matplotlib's name)
        ca, fa = alpha
        return orig(self, (1.0 if ca == 0 else ca, 1.0 if fa == 0 else fa))

    backend_pdf.PdfFile.alphaState = alphaState
    try:
        yield
    finally:
        backend_pdf.PdfFile.alphaState = orig


def _texts(data: bytes) -> Iterator[bytes]:
    yield data
    for m in re.finditer(rb"/Type\s*/ObjStm.*?stream\r?\n(.*?)endstream", data, re.S):
        try:
            yield zlib.decompress(m.group(1))
        except Exception:      # pragma: no cover - malformed stream, ignore
            pass


def scan_transparency(path: Path | str) -> dict[str, int]:
    """Count PDF/A-1 transparency features: /SMask (not /None), /CA or /ca below 1, transparency groups,
    non-Normal blend modes. An empty dict means clean."""
    data = Path(path).read_bytes()
    res: collections.Counter[str] = collections.Counter()
    for t in _texts(data):
        for m in re.finditer(rb"/SMask\s*(/None|\d+\s+\d+\s+R|<<)", t):
            if m.group(1) != b"/None":
                res["SMask"] += 1
        for key in (b"CA", b"ca"):
            for m in re.finditer(rb"/" + key + rb"\s+([0-9.]+)(?![0-9.])", t):
                if float(m.group(1)) < 1:
                    res["/" + key.decode() + " <1"] += 1
        res["Group/Transparency"] += len(re.findall(rb"/Group\s*<<[^>]*?/S\s*/Transparency", t))
        for m in re.finditer(rb"/BM\s*(\[[^\]]*\]|/\w+)", t):
            for name in re.findall(rb"/(\w+)", m.group(1)):
                if name not in (b"Normal", b"Compatible"):
                    res["BM " + name.decode()] += 1
    objs = _objects(data)                    # transparency groups given as indirect references
    for m in re.finditer(rb"/Group\s+(\d+)\s+0\s+R", data):
        body = objs.get(int(m.group(1)), b"")
        if re.search(rb"/S\s*/Transparency", body):
            res["Group/Transparency"] += 1
    return {k: v for k, v in res.items() if v}


def _objects(data: bytes) -> dict[int, bytes]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(rb"(?m)^(\d+) 0 obj\n(.*?)\nendobj", data, re.S)}


def _stream(body: bytes) -> bytes:
    m = re.search(rb"stream\r?\n(.*)\r?\nendstream", body, re.S)
    raw = m.group(1)
    return zlib.decompress(raw) if b"/FlateDecode" in body[:m.start()] else raw


def fonts_missing_cidset(path: Path | str) -> int:
    """Number of CIDFontType2 fonts whose descriptor has no /CIDSet (0 for a PDF/A-1b-ready figure)."""
    objs = _objects(Path(path).read_bytes())
    n = 0
    for body in objs.values():
        if b"/Subtype /CIDFontType2" in body:
            fd = int(re.search(rb"/FontDescriptor (\d+) 0 R", body).group(1))
            n += b"/CIDSet" not in objs[fd]
    return n


def add_cidset(path: Path | str, out: Path | str | None = None, replace_incomplete: bool = False) -> int:
    """Add the /CIDSet streams PDF/A-1 requires to every CIDFontType2 subset (incremental update; the original
    bytes are kept). Writes to ``out`` (default: in place). Returns the number of fonts patched (0 = nothing to do).

    The CIDSet lists every CID covered by the font's CIDToGIDMap: each maps to a glyph that exists in the embedded
    program (unused codes map to GID 0, .notdef). This is what PDF/A-2 (ISO 19005-2, 6.2.11.4.2) requires of a
    CIDSet that is present ("all CIDs present in the font program"), and it satisfies PDF/A-1 (6.3.5) as well.
    veraPDF 1.30.0 accepts it under both 1b and 2b; a set listing only the CIDs with a non-zero GID passes 1b but
    fails 2b (checked 2026-10-13).
    """
    path = Path(path)
    data = path.read_bytes()
    objs = _objects(data)
    updates: dict[int, bytes] = {}
    new_objs: dict[int, bytes] = {}
    next_num = max(objs) + 1
    for num, body in sorted(objs.items()):
        if b"/Subtype /CIDFontType2" not in body:
            continue
        fd = int(re.search(rb"/FontDescriptor (\d+) 0 R", body).group(1))
        c2g = re.search(rb"/CIDToGIDMap (\d+) 0 R", body)
        if not c2g:
            raise ValueError(f"object {num}: CIDFontType2 without a CIDToGIDMap stream")
        gidmap = _stream(objs[int(c2g.group(1))])
        n_cid = len(gidmap) // 2
        bits = bytearray((n_cid + 7) // 8)
        for cid in range(n_cid):              # every CID the map covers resolves to a glyph of the program
            bits[cid // 8] |= 0x80 >> (cid % 8)
        payload = bytes(bits)
        desc = objs[fd]
        existing = re.search(rb"/CIDSet (\d+) 0 R", desc)
        if existing:
            if not replace_incomplete or _stream(objs[int(existing.group(1))]) == payload:
                continue
            desc = re.sub(rb"/CIDSet \d+ 0 R", b"", desc)       # an incomplete set (fails PDF/A-2b): rebuild it
        new_desc = re.sub(rb">>\s*$", b"/CIDSet %d 0 R >>" % next_num, desc.rstrip())
        updates[fd] = new_desc
        new_objs[next_num] = b"<< /Length %d >>\nstream\n" % len(payload) + payload + b"\nendstream"
        next_num += 1
    target = Path(out) if out is not None else path
    if not updates:
        if target != path:
            target.write_bytes(data)
        return 0
    buf = bytearray(data)
    if not buf.endswith(b"\n"):
        buf += b"\n"
    offsets = {}
    for num in sorted({**updates, **new_objs}):
        offsets[num] = len(buf)
        body = updates.get(num, new_objs.get(num))
        buf += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref_pos = len(buf)
    prev = int(re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data).group(1))
    trailer = re.search(rb"trailer\s*<<(.*?)>>\s*startxref", data, re.S).group(1)
    trailer = re.sub(rb"/Size \d+", b"/Size %d" % next_num, trailer)
    buf += b"xref\n"
    for num in sorted(offsets):
        buf += b"%d 1\n%010d 00000 n \n" % (num, offsets[num])
    buf += b"trailer\n<<" + trailer.strip() + b" /Prev %d >>\nstartxref\n%d\n%%%%EOF\n" % (prev, xref_pos)
    target.write_bytes(bytes(buf))
    return len(updates)


def make_pdfa1b_ready(path: Path | str) -> dict[str, int]:
    """Add CIDSets in place and confirm no transparency remains. Returns a small report; raises if not clean."""
    n = add_cidset(path)
    tr = scan_transparency(path)
    missing = fonts_missing_cidset(path)
    if tr or missing:
        raise RuntimeError(f"{path}: not PDF/A-1b ready: transparency {tr}, fonts without CIDSet {missing}")
    return {"cidset_added": n, "transparency": 0, "fonts_missing_cidset": 0}


def cidsets_complete(path: Path | str) -> bool:
    """True if every CIDFontType2 has a CIDSet listing exactly the CIDs its CIDToGIDMap covers (see add_cidset)."""
    objs = _objects(Path(path).read_bytes())
    for body in objs.values():
        if b"/Subtype /CIDFontType2" not in body:
            continue
        fd = objs[int(re.search(rb"/FontDescriptor (\d+) 0 R", body).group(1))]
        m = re.search(rb"/CIDSet (\d+) 0 R", fd)
        c2g = re.search(rb"/CIDToGIDMap (\d+) 0 R", body)
        if not m or not c2g:
            return False
        n_cid = len(_stream(objs[int(c2g.group(1))])) // 2
        expected = bytearray((n_cid + 7) // 8)
        for cid in range(n_cid):
            expected[cid // 8] |= 0x80 >> (cid % 8)
        if _stream(objs[int(m.group(1))]) != bytes(expected):
            return False
    return True
