"""Turns an uploaded blog PDF into one PNG per page.

This is what makes "a PDF" behave like "a carousel": the reader never opens a PDF
viewer, they just see a sequence of images. ``pypdfium2`` bundles its own PDFium
build, so this needs no system package (poppler, ghostscript) in the API's Docker
image.
"""

import io
from typing import List

import pypdfium2 as pdfium

# Wide enough to stay sharp on a large monitor; a phone downscales it for free.
# Capped rather than "native resolution" because a print-quality export (A0 at
# 300dpi) would otherwise produce a many-megabyte PNG per slide.
TARGET_WIDTH_PX = 1600

# A slide deck export can run long; a page count past this is almost certainly a
# mis-upload (a full report, not a carousel), and rasterizing hundreds of pages
# would make one upload request block for minutes.
MAX_PAGES = 60


class TooManyPages(ValueError):
    def __init__(self, count: int):
        super().__init__(f"PDF has {count} pages; the limit is {MAX_PAGES}.")
        self.count = count


def rasterize_pdf(payload: bytes, target_width_px: int = TARGET_WIDTH_PX) -> List[bytes]:
    """One PNG per page, in order. Raises :class:`TooManyPages` past the cap."""
    pdf = pdfium.PdfDocument(payload)
    try:
        page_count = len(pdf)
        if page_count > MAX_PAGES:
            raise TooManyPages(page_count)

        pages: List[bytes] = []
        for index in range(page_count):
            page = pdf[index]
            width_pt, _height_pt = page.get_size()
            scale = target_width_px / width_pt if width_pt else 2.0
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        return pages
    finally:
        pdf.close()
