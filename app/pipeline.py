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
import time
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
    min_conf : float
        Minimum confidence threshold for filtering results
    detect_headings : bool
        Whether to apply heading detection

    Returns
    -------
    List[str]
        The recognised text lines. Lines with no recognised text are
        filtered out.
    """
    start_time = time.time()
    logger.info(f"Starting OCR on image bytes (size: {len(image_bytes)} bytes)")
    
    if not engine.available:
        error_msg = "OCR engine unavailable"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        # Load and convert image
        load_start = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(image)
        load_elapsed = time.time() - load_start
        logger.info(f"Image loaded and converted to array in {load_elapsed:.3f}s - shape: {arr.shape}")
        
        # Run OCR
        ocr_start = time.time()
        results = engine.image_to_lines(arr)
        ocr_elapsed = time.time() - ocr_start
        logger.info(f"OCR completed in {ocr_elapsed:.3f}s - got {len(results)} raw results")
        
        # Filter by confidence
        filter_start = time.time()
        filtered = [
            text.strip()
            for text, conf in results
            if text.strip() and conf >= min_conf
        ]
        filtered_count = len(results) - len(filtered)
        filter_elapsed = time.time() - filter_start
        logger.info(f"Filtered {filtered_count} lines below confidence {min_conf} in {filter_elapsed:.3f}s")
        
        # Apply heading detection if requested
        if detect_headings:
            heading_start = time.time()
            processed = _DETECT_HEADINGS_FN(filtered)
            heading_elapsed = time.time() - heading_start
            logger.info(f"Heading detection completed in {heading_elapsed:.3f}s")
            result = processed
        else:
            result = filtered
        
        total_elapsed = time.time() - start_time
        logger.info(f"Total ocr_image_bytes processing time: {total_elapsed:.3f}s - returned {len(result)} lines")
        return result
        
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"Error in ocr_image_bytes after {elapsed:.3f}s: {exc}", exc_info=True)
        raise


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
    dpi : int
        DPI for rasterization when OCR is needed
    min_conf : float
        Minimum confidence threshold
    detect_headings : bool
        Whether to detect headings
    force_ocr : bool
        Force OCR even if text layer exists
    select_pages : Optional[List[int]]
        Specific page indices to process (None = all pages)

    Returns
    -------
    List[str]
        A list of extracted page texts with heading markers applied.
    """
    start_time = time.time()
    logger.info(f"Starting PDF processing (size: {len(file_bytes)} bytes, force_ocr: {force_ocr}, dpi: {dpi})")
    
    if force_ocr and not engine.available:
        logger.warning("Force OCR requested but PaddleOCR unavailable; falling back to text extraction")
        force_ocr = False

    try:
        # Open PDF document
        open_start = time.time()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        open_elapsed = time.time() - open_start
        total_pages = len(doc)
        logger.info(f"PDF opened in {open_elapsed:.3f}s - total pages: {total_pages}")
        
        page_texts: List[str] = []
        page_indices = (
            [idx for idx in select_pages if 0 <= idx < len(doc)]
            if select_pages
            else list(range(len(doc)))
        )
        logger.info(f"Processing {len(page_indices)} of {total_pages} pages: {page_indices}")
        
        scale = max(dpi, 72) / 72 if dpi > 0 else 1.0
        matrix = fitz.Matrix(scale, scale)
        
        text_extraction_time = 0.0
        ocr_time = 0.0
        pages_via_text = 0
        pages_via_ocr = 0
        
        for page_num, page_index in enumerate(page_indices, 1):
            page_start = time.time()
            logger.info(f"Processing page {page_num}/{len(page_indices)} (PDF page {page_index + 1})")
            
            try:
                page = doc.load_page(page_index)
                
                # Try to extract text using built-in text layer
                extract_start = time.time()
                text = "" if force_ocr else page.get_text().strip()
                extract_elapsed = time.time() - extract_start
                
                if text and not force_ocr:
                    # Split into lines for heading detection
                    text_extraction_time += extract_elapsed
                    pages_via_text += 1
                    lines = text.splitlines()
                    logger.debug(f"Page {page_index + 1}: Extracted {len(lines)} lines via text layer in {extract_elapsed:.3f}s")
                    
                    heading_start = time.time()
                    processed = _DETECT_HEADINGS_FN(lines) if detect_headings else [ln.strip() for ln in lines]
                    heading_elapsed = time.time() - heading_start
                    page_text = "\n".join(processed)
                    logger.info(f"Page {page_index + 1}: Text extraction completed in {extract_elapsed:.3f}s (heading detection: {heading_elapsed:.3f}s)")
                else:
                    # If no text layer, fallback to rasterisation and OCR
                    pages_via_ocr += 1
                    raster_start = time.time()
                    pix = page.get_pixmap(alpha=False, matrix=matrix)
                    img_data = pix.tobytes("png")
                    raster_elapsed = time.time() - raster_start
                    logger.info(f"Page {page_index + 1}: Rasterized to {len(img_data)} bytes in {raster_elapsed:.3f}s")
                    
                    ocr_start = time.time()
                    processed = ocr_image_bytes(
                        engine,
                        img_data,
                        min_conf=min_conf,
                        detect_headings=detect_headings,
                    )
                    page_ocr_elapsed = time.time() - ocr_start
                    ocr_time += page_ocr_elapsed
                    page_text = "\n".join(processed)
                    logger.info(f"Page {page_index + 1}: OCR completed in {page_ocr_elapsed:.3f}s - extracted {len(processed)} lines")
                
                page_texts.append(page_text)
                page_elapsed = time.time() - page_start
                logger.info(f"Page {page_index + 1}: Total processing time: {page_elapsed:.3f}s")
                
            except Exception as exc:
                page_elapsed = time.time() - page_start
                logger.error(f"Error processing page {page_index + 1} after {page_elapsed:.3f}s: {exc}", exc_info=True)
                # Append empty text on error to maintain page count
                page_texts.append("")
                continue
        
        total_elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"PDF processing completed in {total_elapsed:.3f}s")
        logger.info(f"  Pages processed: {len(page_texts)}")
        logger.info(f"  Pages via text extraction: {pages_via_text} (total: {text_extraction_time:.3f}s)")
        logger.info(f"  Pages via OCR: {pages_via_ocr} (total: {ocr_time:.3f}s)")
        logger.info("=" * 80)
        
        return page_texts
        
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"Fatal error in pdf_to_pages after {elapsed:.3f}s: {exc}", exc_info=True)
        raise


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
    min_conf : float
        Minimum confidence threshold
    detect_headings : bool
        Whether to detect headings

    Returns
    -------
    str
        Recognised text (with heading markers) joined by newlines.
    """
    start_time = time.time()
    logger.info(f"Starting image_to_text processing (image size: {len(file_bytes)} bytes)")
    
    if not engine.available:
        error_msg = "OCR engine unavailable"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        lines = ocr_image_bytes(
            engine,
            file_bytes,
            min_conf=min_conf,
            detect_headings=detect_headings,
        )
        result = "\n".join(lines)
        
        total_elapsed = time.time() - start_time
        logger.info(f"image_to_text completed in {total_elapsed:.3f}s - returned {len(lines)} lines, {len(result)} characters")
        return result
        
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"Error in image_to_text after {elapsed:.3f}s: {exc}", exc_info=True)
        raise


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
