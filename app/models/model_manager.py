import os
import logging
from typing import Dict, List, Any, Optional
import torch
from ultralytics import YOLO

logger = logging.getLogger("ai-model-server")

class ModelManager:
    """Manages AI models for the server"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {
            "yolov8n": {"type": "detection", "path": "yolov8n.pt"},
            "yolov8s": {"type": "detection", "path": "yolov8s.pt"},
            "yolov8m": {"type": "detection", "path": "yolov8m.pt"},
            "yolov8n-seg": {"type": "segmentation", "path": "yolov8n-seg.pt"},
            "yolov8s-seg": {"type": "segmentation", "path": "yolov8s-seg.pt"},
        }
        self.default_model = "yolov8n"
        self.models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load default model on startup
        self.load_model(self.default_model)
        
    def load_model(self, model_name: str) -> Any:
        """Load a model by name"""
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configurations")
        
        try:
            logger.info(f"Loading model: {model_name}")
            # For YOLOv8 models, we can load directly from Ultralytics
            model = YOLO(self.model_configs[model_name]["path"])
            self.models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def get_model(self, model_name: str = None) -> Any:
        """Get a model by name, loading it if necessary"""
        if model_name is None:
            model_name = self.default_model
            
        if model_name not in self.models:
            return self.load_model(model_name)
        
        return self.models[model_name]
    
    def list_models(self) -> List[Dict[str, str]]:
        """List all available models"""
        return [
            {"name": name, "type": config["type"], "loaded": name in self.models}
            for name, config in self.model_configs.items()
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.model_configs[model_name]
        return {
            "name": model_name,
            "type": config["type"],
            "loaded": model_name in self.models,
            "path": config["path"]
        }
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self.models:
            try:
                # Remove model from dictionary to free memory
                del self.models[model_name]
                # Force garbage collection
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return True
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {str(e)}")
                return False
        return False