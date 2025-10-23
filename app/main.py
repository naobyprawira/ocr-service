# app/main.py
from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.schemas import OcrLikeResponse
from app.pipeline import (
    pdf_to_pages,
    image_to_text,
    build_response_payload,
)
from app.ocr_engine import OcrEngine

app = FastAPI(title="Paddle OCR Service", version="1.2.0")


def get_engine(lang: str = "en") -> OcrEngine:
    engine: OcrEngine | None = getattr(get_engine, "_engine", None)  # type: ignore[attr-defined]
    if engine is None or engine.lang != lang or not engine.available:
        engine = OcrEngine(lang=lang)
        get_engine._engine = engine  # type: ignore[attr-defined]
    return engine  # type: ignore[return-value,attr-defined]


def _parse_page_range(spec: str | None) -> Optional[list[int]]:
    if not spec or spec.lower() == "all":
        return None
    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start, end = int(a) - 1, int(b) - 1
            if start > end or start < 0:
                continue
            pages.update(range(start, end + 1))
        else:
            idx = int(chunk) - 1
            if idx >= 0:
                pages.add(idx)
    return sorted(pages)


@app.post("/parse", response_model=OcrLikeResponse)
async def parse_pdf(
    file: UploadFile = File(..., description="PDF file"),
    lang: str = Form(default="en"),
    page_range: str = Form(default="all"),
    dpi: int = Form(default=300),
    min_confidence: float = Form(default=0.5),
    detect_headings: bool = Form(default=True),
    force_ocr: bool = Form(default=False),
) -> JSONResponse:
    if file.content_type not in ("application/pdf", "application/x-pdf", "application/acrobat"):
        raise HTTPException(status_code=415, detail="File must be a PDF")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    engine = get_engine(lang=lang)
    selected_pages = _parse_page_range(page_range)

    try:
        pages_text = pdf_to_pages(
            engine=engine,
            file_bytes=data,
            dpi=dpi,
            min_conf=min_confidence,
            detect_headings=detect_headings,
            force_ocr=force_ocr,          # /parse respects embedded text unless forced
            select_pages=selected_pages,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse failed: {exc}") from exc

    return JSONResponse(content=build_response_payload(pages_text))


@app.post("/ocr", response_model=OcrLikeResponse)
async def ocr_endpoint(
    file: UploadFile = File(..., description="Image or PDF"),
    lang: str = Form(default="en"),
    page_range: str = Form(default="all"),
    dpi: int = Form(default=300),
    min_confidence: float = Form(default=0.5),
    detect_headings: bool = Form(default=True),
    force_ocr: bool = Form(default=True),   # /ocr defaults to OCR path on PDFs
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    ct = (file.content_type or "").lower()
    engine = get_engine(lang=lang)
    selected_pages = _parse_page_range(page_range)

    try:
        if "pdf" in ct:
            pages_text = pdf_to_pages(
                engine=engine,
                file_bytes=data,
                dpi=dpi,
                min_conf=min_confidence,
                detect_headings=detect_headings,
                force_ocr=force_ocr,       # default True for /ocr
                select_pages=selected_pages,
            )
        elif ct.startswith("image/"):
            page_text = image_to_text(
                engine,
                data,
                min_conf=min_confidence,
                detect_headings=detect_headings,
            )
            pages_text = [page_text]
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    return JSONResponse(content=build_response_payload(pages_text))
