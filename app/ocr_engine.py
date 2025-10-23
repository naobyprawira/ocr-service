"""PaddleOCR engine wrapper used by the FastAPI service."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import error path
    _PADDLE_AVAILABLE = False
    PaddleOCR = None  # type: ignore
    logger.warning("PaddleOCR import failed: %s", exc)


class OcrEngine:
    """Wraps PaddleOCR to expose a simple `image_to_lines` helper."""

    def __init__(self, lang: str = "latin") -> None:
        self.lang = lang
        self.available = _PADDLE_AVAILABLE
        self.ocr: PaddleOCR | None

        if not self.available:
            self.ocr = None
            return

        try:
            # `use_angle_cls` remains for backwards compatibility; newer
            # releases ignore it in favour of `use_textline_orientation`.
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception as exc:
            logger.error("PaddleOCR initialisation failed: %s", exc)
            self.available = False
            self.ocr = None

    def _call_ocr(self, image: np.ndarray):
        """Invoke PaddleOCR, handling signature differences across versions."""
        if not self.available or self.ocr is None:
            raise RuntimeError("PaddleOCR engine is unavailable")

        try:
            return self.ocr.ocr(image, cls=True)
        except TypeError as exc:
            if "cls" in str(exc):
                logger.info("Retrying PaddleOCR inference without cls flag")
                return self.ocr.ocr(image)
            raise

    def image_to_lines(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Return recognised text lines along with confidence scores.
        
        Supports both old (list-based) and new (OCRResult-based) PaddleOCR APIs.
        """
        try:
            pages = self._call_ocr(image)
        except Exception as exc:
            logger.error("PaddleOCR inference failed: %s", exc)
            return []

        lines: List[Tuple[str, float]] = []
        
        # Handle new PaddleOCR API (returns OCRResult objects)
        for page in pages or []:
            # New API: OCRResult objects have .json or .str properties
            if hasattr(page, 'json'):
                result_dict = page.json if isinstance(page.json, dict) else page.str
                # The actual data is nested under 'res' key
                if 'res' in result_dict:
                    result_dict = result_dict['res']
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                for text, conf in zip(rec_texts, rec_scores):
                    if isinstance(text, str) and text.strip():
                        lines.append((text, float(conf)))
            # Old API: list-based structure
            elif isinstance(page, list):
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
        
        return lines
