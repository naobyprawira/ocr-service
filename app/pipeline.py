"""
Document processing pipeline for the OCR service.

This module contains functions to extract plain text from uploaded
documents (images or PDFs), apply simple heuristics to detect
headings, and assemble the final response payload expected by the
FastAPI endpoints. The pipeline hides all details of PDF parsing and
image OCR behind a clear API.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Union

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .ocr_engine import OcrEngine

logger = logging.getLogger(__name__)


def detect_headings(lines: List[str]) -> List[str]:
    """Apply heuristic rules to identify heading lines.

    A line is considered a heading if it meets the following criteria:
    - It is relatively short (fewer than 8 words)
    - It is either fully capitalised or title cased
    - It is surrounded by blank lines (before or after)

    Parameters
    ----------
    lines : List[str]
        The list of text lines extracted from OCR.

    Returns
    -------
    List[str]
        A new list of lines where each heading line is prefixed with
        "## ". Non-heading lines are left unchanged.
    """
    result: List[str] = []
    num_lines = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Determine if blank lines exist around this line
        prev_blank = idx == 0 or not lines[idx - 1].strip()
        next_blank = idx == num_lines - 1 or not lines[idx + 1].strip()

        # Simple heuristics for heading detection
        words = stripped.split()
        is_short = len(words) <= 8  # short phrases
        is_title_case = stripped.istitle()  # capitalises first letter of each word
        is_upper = stripped.isupper()  # all uppercase
        if stripped and is_short and prev_blank and next_blank and (is_title_case or is_upper):
            result.append("## " + stripped)
        else:
            result.append(stripped)
    return result


_DETECT_HEADINGS_FN = detect_headings


def ocr_image_bytes(
    engine: OcrEngine,
    image_bytes: bytes,
    *,
    min_conf: float = 0.0,
    detect_headings: bool = True,
) -> List[str]:
    """Run OCR on image bytes and return recognised lines of text.

    Parameters
    ----------
    engine : OcrEngine
        The OCR engine instance used to perform inference.
    image_bytes : bytes
        Encoded image data (PNG or JPEG) used as input.

    Returns
    -------
    List[str]
        The recognised text lines. Lines with no recognised text are
        filtered out.
    """
    if not engine.available:
        raise RuntimeError("OCR engine unavailable")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    results = engine.image_to_lines(arr)
    filtered = [
        text.strip()
        for text, conf in results
        if text.strip() and conf >= min_conf
    ]
    if not detect_headings:
        return filtered
    return _DETECT_HEADINGS_FN(filtered)


def pdf_to_pages(
    engine: OcrEngine,
    file_bytes: bytes,
    *,
    dpi: int = 300,
    min_conf: float = 0.0,
    detect_headings: bool = True,
    force_ocr: bool = False,
    select_pages: Optional[List[int]] = None,
) -> List[str]:
    """Extract text from each page of a PDF.

    This implementation uses PyMuPDF's built-in text extraction
    capabilities instead of OCR. It falls back to rasterisation via
    `get_pixmap` and stub OCR only if the extracted text is empty.
    Heading markers are applied to the extracted lines.

    Parameters
    ----------
    engine : OcrEngine
        The OCR engine instance (unused in this stub except for API
        compatibility).
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    List[str]
        A list of extracted page texts with heading markers applied.
    """
    if force_ocr and not engine.available:
        logger.warning("Force OCR requested but PaddleOCR unavailable; falling back to text extraction")
        force_ocr = False

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts: List[str] = []
    page_indices = (
        [idx for idx in select_pages if 0 <= idx < len(doc)]
        if select_pages
        else list(range(len(doc)))
    )
    scale = max(dpi, 72) / 72 if dpi > 0 else 1.0
    matrix = fitz.Matrix(scale, scale)
    for page_index in page_indices:
        page = doc.load_page(page_index)
        # Try to extract text using built-in text layer
        text = "" if force_ocr else page.get_text().strip()
        if text and not force_ocr:
            # Split into lines for heading detection
            lines = text.splitlines()
            processed = _DETECT_HEADINGS_FN(lines) if detect_headings else [ln.strip() for ln in lines]
            page_text = "\n".join(processed)
        else:
            # If no text layer, fallback to rasterisation and stub OCR
            pix = page.get_pixmap(alpha=False, matrix=matrix)
            img_data = pix.tobytes("png")
            processed = ocr_image_bytes(
                engine,
                img_data,
                min_conf=min_conf,
                detect_headings=detect_headings,
            )
            page_text = "\n".join(processed)
        page_texts.append(page_text)
    return page_texts


def image_to_text(
    engine: OcrEngine,
    file_bytes: bytes,
    *,
    min_conf: float = 0.0,
    detect_headings: bool = True,
) -> str:
    """Extract text from a single image.

    Parameters
    ----------
    engine : OcrEngine
        The OCR engine instance.
    file_bytes : bytes
        Raw bytes of the image file.

    Returns
    -------
    str
        Recognised text (with heading markers) joined by newlines.
    """
    if not engine.available:
        raise RuntimeError("OCR engine unavailable")

    lines = ocr_image_bytes(
        engine,
        file_bytes,
        min_conf=min_conf,
        detect_headings=detect_headings,
    )
    return "\n".join(lines)


def build_response(page_texts: List[str]) -> Dict[str, Union[int, List[Dict[str, Union[int, str]]], str]]:
    """Build the unified response structure for OCR endpoints.

    Parameters
    ----------
    page_texts : List[str]
        List of text for each page.

    Returns
    -------
    dict
        Response dictionary matching the expected schema.
    """
    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({"index": idx, "text": text})
    # Concatenate pages with separators
    concatenated = ""
    for i, text in enumerate(page_texts):
        if i > 0:
            concatenated += f"\n\n===== Page {i} =====\n\n"
        concatenated += text
    return {
        "doc": {"pages": len(page_texts)},
        "pages": pages,
        "text": concatenated,
    }


def build_response_payload(page_texts: List[str]) -> Dict[str, Union[int, List[Dict[str, Union[int, str]]], str]]:
    """Backward-compatible alias used by the FastAPI layer."""
    return build_response(page_texts)
