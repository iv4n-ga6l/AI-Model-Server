import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import shutil

logger = logging.getLogger("ai-model-server")

class ServiceManager:
    """Manages the AI Model Server services and deployment"""
    
    def __init__(self, model_manager, task_scheduler):
        self.model_manager = model_manager
        self.task_scheduler = task_scheduler
        self.config_path = os.path.join(os.getcwd(), "config", "service_config.json")
        self.service_config = self._load_config()
        
        # Schedule periodic tasks
        self._schedule_maintenance_tasks()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration from file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading service config: {str(e)}")
        
        # Default configuration
        default_config = {
            "max_concurrent_inferences": 4,
            "cleanup_interval_hours": 24,
            "max_result_age_days": 7,
            "default_backend": "cpu",  # or "gpu"
            "model_cache_size": 2,  # Number of models to keep in memory
            "last_updated": datetime.now().isoformat()
        }
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save service configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving service config: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update service configuration"""
        config = self.service_config.copy()
        config.update(updates)
        config["last_updated"] = datetime.now().isoformat()
        
        self._save_config(config)
        self.service_config = config
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current service configuration"""
        return self.service_config
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(os.getcwd())
        
        gpu_info = self._get_gpu_info()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            },
            "gpu": gpu_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "available": True,
                    "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    "memory_allocated": torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else None,
                    "memory_reserved": torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else None
                }
            else:
                return {"available": False}
        except ImportError:
            return {"available": False, "error": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _schedule_maintenance_tasks(self) -> None:
        """Schedule periodic maintenance tasks"""
        # Schedule cleanup task
        cleanup_interval = timedelta(hours=self.service_config.get("cleanup_interval_hours", 24))
        self.task_scheduler.schedule_periodic_task(
            task_id="cleanup_old_results",
            interval=cleanup_interval,
            coroutine_factory=self._cleanup_old_results,
            description="Clean up old prediction results"
        )
        
        # Schedule model cache management
        self.task_scheduler.schedule_periodic_task(
            task_id="manage_model_cache",
            interval=timedelta(hours=1),
            coroutine_factory=self._manage_model_cache,
            description="Manage model cache"
        )
    
    async def _cleanup_old_results(self) -> Dict[str, Any]:
        """Clean up old prediction results"""
        max_age_days = self.service_config.get("max_result_age_days", 7)
        max_age = timedelta(days=max_age_days)
        now = datetime.now()
        
        results_dir = os.path.join(os.getcwd(), "static", "results")
        uploads_dir = os.path.join(os.getcwd(), "static", "uploads")
        
        deleted_count = 0
        
        # Process results directory
        if os.path.exists(results_dir):
            for prediction_id in os.listdir(results_dir):
                result_path = os.path.join(results_dir, prediction_id)
                if os.path.isdir(result_path):
                    # Check the modification time of the directory
                    mod_time = datetime.fromtimestamp(os.path.getmtime(result_path))
                    if now - mod_time > max_age:
                        try:
                            shutil.rmtree(result_path)
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting old result {prediction_id}: {str(e)}")
        
        # Process uploads directory
        if os.path.exists(uploads_dir):
            for prediction_id in os.listdir(uploads_dir):
                upload_path = os.path.join(uploads_dir, prediction_id)
                if os.path.isdir(upload_path):
                    # Check the modification time of the directory
                    mod_time = datetime.fromtimestamp(os.path.getmtime(upload_path))
                    if now - mod_time > max_age:
                        try:
                            shutil.rmtree(upload_path)
                        except Exception as e:
                            logger.error(f"Error deleting old upload {prediction_id}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old prediction results")
        return {"deleted_count": deleted_count}
    
    async def _manage_model_cache(self) -> Dict[str, Any]:
        """Manage model cache to limit memory usage"""
        max_models = self.service_config.get("model_cache_size", 2)
        loaded_models = self.model_manager.list_models()
        
        # Count loaded models
        loaded_count = sum(1 for model in loaded_models if model["loaded"])
        
        # If we're under the limit, do nothing
        if loaded_count <= max_models:
            return {"status": "ok", "loaded_models": loaded_count, "max_models": max_models}
        
        # Unload excess models
        models_to_unload = loaded_count - max_models
        unloaded = 0
        
        # Keep the default model loaded
        default_model = self.model_manager.default_model
        
        for model in loaded_models:
            if unloaded >= models_to_unload:
                break
                
            if model["loaded"] and model["name"] != default_model:
                success = self.model_manager.unload_model(model["name"])
                if success:
                    unloaded += 1
        
        logger.info(f"Unloaded {unloaded} models to manage cache")
        return {"status": "ok", "unloaded_models": unloaded, "max_models": max_models}
    
    def deploy_model(self, model_name: str, backend: str = None) -> Dict[str, Any]:
        """Deploy a model with specified backend"""
        if backend is None:
            backend = self.service_config.get("default_backend", "cpu")
        
        # Validate backend
        if backend not in ["cpu", "gpu"]:
            raise ValueError(f"Invalid backend: {backend}. Must be 'cpu' or 'gpu'")
        
        # Check if GPU is available for GPU backend
        if backend == "gpu":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("GPU backend requested but no GPU is available")
        
        # Load the model
        model = self.model_manager.load_model(model_name)
        
        # Set the device
        if hasattr(model, "to"):
            device = "cuda:0" if backend == "gpu" else "cpu"
            model.to(device)
        
        return {
            "model": model_name,
            "backend": backend,
            "status": "deployed",
            "timestamp": datetime.now().isoformat()
        }