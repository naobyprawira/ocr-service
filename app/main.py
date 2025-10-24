# app/main.py
from __future__ import annotations

import io
import logging
import time
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
from app.logging_config import setup_logging, detect_paddle_device
from app.rate_limiter import init_rate_limiter, get_rate_limiter

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Paddle OCR Service", version="1.2.0")

# Global engine cache
_engine_cache = {}


def get_engine(lang: str = "en") -> OcrEngine:
    """Get or create OCR engine instance with caching and error handling."""
    global _engine_cache
    
    # Check cache
    if lang in _engine_cache:
        engine = _engine_cache[lang]
        if engine.available:
            return engine
        else:
            logger.warning(f"Cached engine for '{lang}' is unavailable, reinitializing...")
    
    # Create new engine
    try:
        logger.info(f"Creating new OCR engine for language: {lang}")
        engine = OcrEngine(lang=lang)
        
        if not engine.available:
            logger.error(f"Failed to initialize OCR engine for '{lang}'")
            # Don't cache failed engine
            raise RuntimeError(f"OCR engine unavailable for language '{lang}'")
        
        _engine_cache[lang] = engine
        logger.info(f"OCR engine for '{lang}' cached successfully")
        return engine
        
    except Exception as exc:
        logger.error(f"Fatal error creating OCR engine: {exc}", exc_info=True)
        raise RuntimeError(f"Cannot initialize OCR engine: {exc}") from exc


@app.on_event("startup")
async def startup_event():
    """Initialize OCR engine on startup to catch errors early."""
    logger.info("=" * 80)
    logger.info("Starting Paddle OCR Service")
    
    # Detect device
    device_type, device_info = detect_paddle_device()
    logger.info(f"🖥️  Running on: {device_type} - {device_info}")
    
    # Initialize rate limiter for concurrent request processing
    init_rate_limiter(
        max_concurrent=3,      # Maximum 3 concurrent operations
        max_queue_size=100,     # Queue up to 100 requests
        timeout=6000.0           # 6000 second timeout for waiting
    )
    logger.info("Request rate limiter initialized")
    
    try:
        # Pre-initialize default engine
        logger.info("Pre-initializing default OCR engine...")
        engine = get_engine("en")
        
        if engine.available:
            logger.info("✅ OCR engine initialized successfully")
            logger.info("Service is ready to accept requests")
        else:
            logger.error("❌ OCR engine initialization failed")
            logger.error("Service will start but OCR requests will fail")
            
    except Exception as exc:
        logger.error(f"❌ Startup error: {exc}", exc_info=True)
        logger.error("Service will start but OCR functionality may not work")
    
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=" * 80)
    logger.info("Shutting down Paddle OCR Service")
    _engine_cache.clear()
    logger.info("Service stopped")
    logger.info("=" * 80)


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
    request_start = time.time()
    logger.info("=" * 80)
    logger.info(f"POST /parse - filename: {file.filename}, content_type: {file.content_type}")
    logger.info(f"Parameters - lang: {lang}, page_range: {page_range}, dpi: {dpi}, min_confidence: {min_confidence}, detect_headings: {detect_headings}, force_ocr: {force_ocr}")
    
    if file.content_type not in ("application/pdf", "application/x-pdf", "application/acrobat"):
        logger.warning(f"Rejected file with unsupported content type: {file.content_type}")
        raise HTTPException(status_code=415, detail="File must be a PDF")

    data = await file.read()
    if not data:
        logger.error("Empty file uploaded")
        raise HTTPException(status_code=400, detail="Empty file")

    logger.info(f"File read successfully - size: {len(data)} bytes")
    
    engine = get_engine(lang=lang)
    selected_pages = _parse_page_range(page_range)
    logger.info(f"Selected pages: {selected_pages if selected_pages else 'all'}")

    try:
        processing_start = time.time()
        
        # Acquire processing access (works with both CPU and GPU)
        limiter = get_rate_limiter()
        async with limiter.acquire():
            pages_text = pdf_to_pages(
                engine=engine,
                file_bytes=data,
                dpi=dpi,
                min_conf=min_confidence,
                detect_headings=detect_headings,
                force_ocr=force_ocr,          # /parse respects embedded text unless forced
                select_pages=selected_pages,
            )
        
        processing_elapsed = time.time() - processing_start
        logger.info(f"PDF processing completed in {processing_elapsed:.3f}s - {len(pages_text)} pages")
        
        response_start = time.time()
        response_content = build_response_payload(pages_text)
        response_elapsed = time.time() - response_start
        logger.info(f"Response payload built in {response_elapsed:.3f}s")
        
        total_elapsed = time.time() - request_start
        logger.info(f"Total /parse request time: {total_elapsed:.3f}s")
        logger.info("=" * 80)
        
        return JSONResponse(content=response_content)
        
    except RuntimeError as exc:
        # Queue full or rate limiter error
        elapsed = time.time() - request_start
        logger.error(f"/parse request rejected after {elapsed:.3f}s: {exc}")
        logger.info("=" * 80)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        # Processing timeout
        elapsed = time.time() - request_start
        logger.error(f"/parse request timeout after {elapsed:.3f}s: {exc}")
        logger.info("=" * 80)
        raise HTTPException(status_code=503, detail="Processing timeout - server is busy") from exc
    except Exception as exc:
        elapsed = time.time() - request_start
        logger.error(f"/parse request failed after {elapsed:.3f}s: {exc}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Parse failed: {exc}") from exc


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
    request_start = time.time()
    logger.info("=" * 80)
    logger.info(f"POST /ocr - filename: {file.filename}, content_type: {file.content_type}")
    logger.info(f"Parameters - lang: {lang}, page_range: {page_range}, dpi: {dpi}, min_confidence: {min_confidence}, detect_headings: {detect_headings}, force_ocr: {force_ocr}")
    
    data = await file.read()
    if not data:
        logger.error("Empty file uploaded")
        raise HTTPException(status_code=400, detail="Empty file")

    logger.info(f"File read successfully - size: {len(data)} bytes")
    
    ct = (file.content_type or "").lower()
    engine = get_engine(lang=lang)
    selected_pages = _parse_page_range(page_range)

    try:
        processing_start = time.time()
        
        # Acquire processing access (works with both CPU and GPU)
        limiter = get_rate_limiter()
        async with limiter.acquire():
            if "pdf" in ct:
                logger.info(f"Processing as PDF - selected pages: {selected_pages if selected_pages else 'all'}")
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
                logger.info(f"Processing as image - content type: {ct}")
                page_text = image_to_text(
                    engine,
                    data,
                    min_conf=min_confidence,
                    detect_headings=detect_headings,
                )
                pages_text = [page_text]
            else:
                logger.warning(f"Unsupported file type: {ct}")
                raise HTTPException(status_code=415, detail="Unsupported file type")
        
        processing_elapsed = time.time() - processing_start
        logger.info(f"OCR processing completed in {processing_elapsed:.3f}s - {len(pages_text)} pages")
        
        response_start = time.time()
        response_content = build_response_payload(pages_text)
        response_elapsed = time.time() - response_start
        logger.info(f"Response payload built in {response_elapsed:.3f}s")
        
        total_elapsed = time.time() - request_start
        logger.info(f"Total /ocr request time: {total_elapsed:.3f}s")
        logger.info("=" * 80)
        
        return JSONResponse(content=response_content)
        
    except HTTPException:
        elapsed = time.time() - request_start
        logger.error(f"/ocr request failed after {elapsed:.3f}s with HTTPException", exc_info=True)
        logger.info("=" * 80)
        raise
    except RuntimeError as exc:
        # Queue full or rate limiter error
        elapsed = time.time() - request_start
        logger.error(f"/ocr request rejected after {elapsed:.3f}s: {exc}")
        logger.info("=" * 80)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        # Processing timeout
        elapsed = time.time() - request_start
        logger.error(f"/ocr request timeout after {elapsed:.3f}s: {exc}")
        logger.info("=" * 80)
        raise HTTPException(status_code=503, detail="Processing timeout - server is busy") from exc
    except Exception as exc:
        elapsed = time.time() - request_start
        logger.error(f"/ocr request failed after {elapsed:.3f}s: {exc}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.get("/health")
async def health_check():
    """Health check endpoint with rate limiter statistics."""
    try:
        limiter = get_rate_limiter()
        stats = limiter.get_stats()
        
        return {
            "status": "healthy",
            "service": "Paddle OCR Service",
            "rate_limiter": stats
        }
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "degraded",
            "service": "Paddle OCR Service",
            "error": str(exc)
        }


@app.get("/stats")
async def get_stats():
    """Get detailed rate limiter statistics."""
    try:
        limiter = get_rate_limiter()
        return limiter.get_stats()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
