"""
Logging configuration for NBA play-by-play data pipeline.

Provides consistent logging setup across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: bool = True,
    log_dir: Path = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to write logs to a file
        log_dir: Directory for log files (default: project_root/logs)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if needed
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("nba_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler (simpler format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    if log_file:
        log_filename = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_filename}")
    
    return logger
 

def get_logger(name: str = "nba_pipeline") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: "nba_pipeline")
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default logging
    if not logger.handlers:
        setup_logging()
        logger = logging.getLogger(name)
    
    return logger
 
