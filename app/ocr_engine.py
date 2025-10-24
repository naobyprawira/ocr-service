"""PaddleOCR engine wrapper used by the FastAPI service."""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Suppress Windows DLL warnings
if sys.platform == 'win32':
    warnings.filterwarnings('ignore', message='.*zlibwapi.dll.*')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflicts

# Check NumPy version compatibility
_NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
if _NUMPY_VERSION >= (2, 0):
    logger.error(
        f"NumPy {np.__version__} is not compatible with PaddlePaddle. "
        "Please downgrade: pip install 'numpy<2.0'"
    )
    _PADDLE_AVAILABLE = False
    PaddleOCR = None
else:
    try:
        # Suppress PaddleOCR debug output
        os.environ['FLAGS_prim_enable_dynamic'] = '1'
        os.environ['FLAGS_enable_pir_api'] = '1'
        
        from paddleocr import PaddleOCR  # type: ignore

        _PADDLE_AVAILABLE = True
        logger.info(f"NumPy {np.__version__} compatible with PaddlePaddle")
    except Exception as exc:  # pragma: no cover - import error path
        _PADDLE_AVAILABLE = False
        PaddleOCR = None  # type: ignore
        logger.error(f"PaddleOCR import failed: {exc}", exc_info=True)


class OcrEngine:
    """Wraps PaddleOCR to expose a simple `image_to_lines` helper."""

    def __init__(self, lang: str = "latin") -> None:
        """Initialize OCR engine with specified language.
        
        Parameters
        ----------
        lang : str
            Language code for OCR (default: "latin")
        """
        start_time = time.time()
        self.lang = lang
        self.available = _PADDLE_AVAILABLE
        self.ocr: PaddleOCR | None = None

        if not self.available:
            logger.warning("PaddleOCR library not available")
            return

        try:
            # `use_angle_cls` remains for backwards compatibility; newer
            # releases ignore it in favour of `use_textline_orientation`.
            logger.info(f"Initializing PaddleOCR engine with language: {lang}")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=True,
                show_log=False,  # Suppress verbose output
                enable_mkldnn=False,  # Disable MKL-DNN to avoid conflicts
                use_tensorrt=False,  # Disable TensorRT
            )
            elapsed = time.time() - start_time
            logger.info(f"PaddleOCR engine initialized successfully in {elapsed:.2f}s")
            self.available = True
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(f"PaddleOCR initialisation failed after {elapsed:.2f}s: {exc}", exc_info=True)
            self.available = False
            self.ocr = None

    def _call_ocr(self, image: np.ndarray):
        """Invoke PaddleOCR, handling signature differences across versions."""
        if not self.available or self.ocr is None:
            error_msg = "PaddleOCR engine is unavailable"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        start_time = time.time()
        try:
            logger.debug(f"Starting OCR inference on image shape: {image.shape}")
            result = self.ocr.ocr(image, cls=True)
            elapsed = time.time() - start_time
            logger.info(f"OCR inference completed in {elapsed:.3f}s")
            return result
        except TypeError as exc:
            if "cls" in str(exc):
                logger.warning("PaddleOCR does not support 'cls' parameter, retrying without it")
                result = self.ocr.ocr(image)
                elapsed = time.time() - start_time
                logger.info(f"OCR inference completed (without cls) in {elapsed:.3f}s")
                return result
            logger.error(f"TypeError during OCR inference: {exc}", exc_info=True)
            raise
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(f"OCR inference failed after {elapsed:.3f}s: {exc}", exc_info=True)
            raise

    def image_to_lines(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Return recognised text lines along with confidence scores.
        
        Supports both old (list-based) and new (OCRResult-based) PaddleOCR APIs.
        
        Parameters
        ----------
        image : np.ndarray
            Input image as numpy array
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (text, confidence) tuples
        """
        start_time = time.time()
        logger.info(f"Starting image_to_lines processing for image shape: {image.shape}")
        
        try:
            pages = self._call_ocr(image)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(f"PaddleOCR inference failed after {elapsed:.3f}s: {exc}", exc_info=True)
            return []

        lines: List[Tuple[str, float]] = []
        parse_start = time.time()
        
        # Handle new PaddleOCR API (returns OCRResult objects)
        page_count = 0
        for page_idx, page in enumerate(pages or []):
            page_count += 1
            page_start = time.time()
            page_lines = 0
            
            try:
                # New API: OCRResult objects have .json or .str properties
                if hasattr(page, 'json'):
                    logger.debug(f"Processing page {page_idx} using new OCRResult API")
                    result_dict = page.json if isinstance(page.json, dict) else page.str
                    # The actual data is nested under 'res' key
                    if 'res' in result_dict:
                        result_dict = result_dict['res']
                    rec_texts = result_dict.get('rec_texts', [])
                    rec_scores = result_dict.get('rec_scores', [])
                    for text, conf in zip(rec_texts, rec_scores):
                        if isinstance(text, str) and text.strip():
                            lines.append((text, float(conf)))
                            page_lines += 1
                # Old API: list-based structure
                elif isinstance(page, list):
                    logger.debug(f"Processing page {page_idx} using old list-based API")
                    for item in page:
                        if (
                            isinstance(item, list)
                            and len(item) >= 2
                            and isinstance(item[1], tuple)
                            and len(item[1]) == 2
                        ):
                            text, conf = item[1]
                            if isinstance(text, str):
                                lines.append((text, float(conf)))
                                page_lines += 1
                
                page_elapsed = time.time() - page_start
                logger.info(f"Page {page_idx} processed in {page_elapsed:.3f}s - extracted {page_lines} lines")
                
            except Exception as exc:
                page_elapsed = time.time() - page_start
                logger.error(f"Error processing page {page_idx} after {page_elapsed:.3f}s: {exc}", exc_info=True)
                continue
        
        parse_elapsed = time.time() - parse_start
        total_elapsed = time.time() - start_time
        
        # Calculate average confidence
        avg_conf = sum(conf for _, conf in lines) / len(lines) if lines else 0.0
        
        logger.info(f"Completed parsing {page_count} page(s) in {parse_elapsed:.3f}s")
        logger.info(f"Total image_to_lines processing time: {total_elapsed:.3f}s")
        logger.info(f"Extracted {len(lines)} total lines with average confidence: {avg_conf:.3f}")
        
        return lines
