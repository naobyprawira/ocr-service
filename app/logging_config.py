"""Logging configuration for the OCR service."""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def detect_paddle_device() -> tuple[str, str]:
    """Detect whether PaddlePaddle is using CPU or GPU.
    
    Returns
    -------
    tuple[str, str]
        (device_type, device_info) where device_type is 'GPU' or 'CPU'
        and device_info contains additional details.
    """
    try:
        import paddle
        
        # Check if GPU is available and enabled
        if paddle.is_compiled_with_cuda():
            if paddle.device.cuda.is_available():
                device_count = paddle.device.cuda.device_count()
                device_name = paddle.device.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                return "GPU", f"NVIDIA CUDA (Devices: {device_count}, {device_name})"
            else:
                return "GPU", "CUDA compiled but not available"
        else:
            return "CPU", "CPU version (no CUDA support)"
    except Exception as exc:
        return "UNKNOWN", f"Could not detect device: {exc}"


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """Configure logging to write to both file and console.
    
    Parameters
    ----------
    log_dir : str
        Directory where log files will be stored.
    log_level : int
        Logging level (default: logging.INFO).
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = log_path / f"ocr_service_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation (max 10MB per file, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)  # Log to console at the specified level
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('ppocr').setLevel(logging.ERROR)  # Suppress PaddleOCR debug messages
    logging.getLogger('paddleocr').setLevel(logging.ERROR)
    logging.getLogger('paddle').setLevel(logging.WARNING)
    
    # Detect device
    device_type, device_info = detect_paddle_device()
    
    # Log initialization
    root_logger.info("=" * 80)
    root_logger.info("OCR Service logging initialized")
    root_logger.info(f"Log file: {log_filename}")
    root_logger.info(f"🖥️  Device: {device_type} - {device_info}")
    root_logger.info("=" * 80)
