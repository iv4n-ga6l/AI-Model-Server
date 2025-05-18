import os
import shutil
from fastapi import UploadFile
from typing import List, Dict, Any
import logging

logger = logging.getLogger("ai-model-server")

async def save_upload_file(upload_file: UploadFile, prediction_id: str) -> str:
    """Save an uploaded file to the uploads directory"""
    uploads_dir = os.path.join(os.getcwd(), "static", "uploads", prediction_id)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Determine file extension
    file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
    file_path = os.path.join(uploads_dir, f"input{file_extension}")
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    logger.info(f"Saved uploaded file to {file_path}")
    return file_path

def create_directories():
    """Create necessary directories for the application"""
    dirs = [
        os.path.join(os.getcwd(), "static"),
        os.path.join(os.getcwd(), "static", "uploads"),
        os.path.join(os.getcwd(), "static", "results"),
        os.path.join(os.getcwd(), "models"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")