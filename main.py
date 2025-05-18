from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import shutil
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

# Import our custom modules
from app.models.model_manager import ModelManager
from app.services.inference_service import InferenceService
from app.services.task_scheduler import TaskScheduler
from app.services.service_manager import ServiceManager
from app.utils.file_utils import save_upload_file, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-model-server")

# Initialize FastAPI app
app = FastAPI(
    title="AI Model Server",
    description="High Performance AI Model Server for Computer Vision Tasks",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
create_directories()

# Initialize model manager, inference service, and task scheduler
model_manager = ModelManager()
inference_service = InferenceService(model_manager)
task_scheduler = TaskScheduler()

# Initialize service manager
service_manager = ServiceManager(model_manager, task_scheduler)

# Start the task scheduler on application startup
@app.on_event("startup")
async def startup_event():
    await task_scheduler.start()
    
# Stop the task scheduler on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await task_scheduler.stop()

# Mount static files directory for serving results
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    return {"message": "Welcome to AI Model Server API", "status": "running"}

@app.get("/models")
async def list_models():
    """List all available models"""
    return {"models": model_manager.list_models()}

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_name: str = "yolov8n", task_type: str = "detection", confidence: float = 0.5):
    """Run inference on uploaded image"""
    try:
        # Generate a unique ID for this prediction
        prediction_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_path = await save_upload_file(file, prediction_id)
        
        # Run inference
        result = await inference_service.run_inference(
            file_path=file_path,
            model_name=model_name,
            task_type=task_type,
            confidence=confidence,
            prediction_id=prediction_id
        )
        
        return {
            "prediction_id": prediction_id,
            "model": model_name,
            "task": task_type,
            "results": result,
            "result_url": f"/static/results/{prediction_id}/result.jpg"
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{prediction_id}")
async def get_prediction_status(prediction_id: str):
    """Get the status of a prediction"""
    status = inference_service.get_prediction_status(prediction_id)
    return {"prediction_id": prediction_id, "status": status}

@app.get("/result/{prediction_id}")
async def get_prediction_result(prediction_id: str):
    """Get the result of a prediction"""
    result_path = f"static/results/{prediction_id}/result.jpg"
    if os.path.exists(result_path):
        return FileResponse(result_path)
    else:
        raise HTTPException(status_code=404, detail="Result not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/tasks")
async def list_tasks():
    """List all scheduled tasks"""
    return task_scheduler.list_tasks()

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a scheduled task"""
    success = task_scheduler.cancel_task(task_id)
    if success:
        return {"message": f"Task {task_id} cancelled successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task"""
    status = task_scheduler.get_task_status(task_id)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    return service_manager.get_system_info()

@app.get("/system/config")
async def get_service_config():
    """Get service configuration"""
    return service_manager.get_config()

@app.post("/system/config")
async def update_service_config(updates: Dict[str, Any]):
    """Update service configuration"""
    return service_manager.update_config(updates)

@app.post("/models/deploy/{model_name}")
async def deploy_model(model_name: str, background_tasks: BackgroundTasks, backend: Optional[str] = None):
    """Deploy a model with specified backend"""
    try:
        # Schedule model deployment as a background task
        task_id = f"deploy_{model_name}_{uuid.uuid4()}"
        background_tasks.add_task(
            task_scheduler.schedule_task,
            task_id=task_id,
            coroutine=service_manager.deploy_model(model_name, backend),
            description=f"Deploy model {model_name} on {backend or 'default'} backend"
        )
        
        return {
            "task_id": task_id,
            "model": model_name,
            "backend": backend or service_manager.get_config().get("default_backend", "cpu"),
            "status": "deploying"
        }
    except Exception as e:
        logger.error(f"Error deploying model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)