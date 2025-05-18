import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger("ai-model-server")

class InferenceService:
    """Service for running inference with AI models"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.active_predictions = {}
        self.results_dir = os.path.join(os.getcwd(), "static", "results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def run_inference(self, file_path: str, model_name: str, task_type: str, confidence: float, prediction_id: str) -> Dict[str, Any]:
        """Run inference on an image file"""
        try:
            # Update prediction status
            self.active_predictions[prediction_id] = "processing"
            
            # Create result directory for this prediction
            result_dir = os.path.join(self.results_dir, prediction_id)
            os.makedirs(result_dir, exist_ok=True)
            
            # Get the model
            model = self.model_manager.get_model(model_name)
            
            # Run inference based on task type
            if task_type == "detection":
                return await self._run_detection(model, file_path, confidence, prediction_id, result_dir)
            elif task_type == "segmentation":
                return await self._run_segmentation(model, file_path, confidence, prediction_id, result_dir)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            self.active_predictions[prediction_id] = "failed"
            raise
        finally:
            # If prediction completed successfully, update status
            if prediction_id in self.active_predictions and self.active_predictions[prediction_id] == "processing":
                self.active_predictions[prediction_id] = "completed"
    
    async def _run_detection(self, model, file_path: str, confidence: float, prediction_id: str, result_dir: str) -> Dict[str, Any]:
        """Run object detection inference"""
        # Run inference asynchronously
        results = await asyncio.to_thread(model, file_path, conf=confidence)
        
        # Process results
        result = results[0]
        
        # Save the visualization
        result_img = result.plot()
        cv2.imwrite(os.path.join(result_dir, "result.jpg"), result_img)
        
        # Extract detection information
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls]
            
            detections.append({
                "class": name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        
        # Save JSON results
        json_result = {
            "prediction_id": prediction_id,
            "model": model.names[0] if hasattr(model, 'names') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        }
        
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(json_result, f, indent=2)
        
        return json_result
    
    async def _run_segmentation(self, model, file_path: str, confidence: float, prediction_id: str, result_dir: str) -> Dict[str, Any]:
        """Run segmentation inference"""
        # Run inference asynchronously
        results = await asyncio.to_thread(model, file_path, conf=confidence)
        
        # Process results
        result = results[0]
        
        # Save the visualization
        result_img = result.plot()
        cv2.imwrite(os.path.join(result_dir, "result.jpg"), result_img)
        
        # Extract segmentation information
        segments = []
        for i, mask in enumerate(result.masks.data):
            if result.boxes is not None and i < len(result.boxes):
                box = result.boxes[i]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                
                # Convert mask to polygon for more compact representation
                contours, _ = cv2.findContours((mask.cpu().numpy() * 255).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                # Get the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Simplify the contour
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    # Convert to list for JSON serialization
                    points = approx.reshape(-1, 2).tolist()
                    
                    segments.append({
                        "class": name,
                        "confidence": conf,
                        "points": points
                    })
        
        # Save JSON results
        json_result = {
            "prediction_id": prediction_id,
            "model": model.names[0] if hasattr(model, 'names') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "segments": segments
        }
        
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(json_result, f, indent=2)
        
        return json_result
    
    def get_prediction_status(self, prediction_id: str) -> str:
        """Get the status of a prediction"""
        if prediction_id not in self.active_predictions:
            # Check if result exists on disk
            result_path = os.path.join(self.results_dir, prediction_id, "result.json")
            if os.path.exists(result_path):
                return "completed"
            return "not_found"
        
        return self.active_predictions[prediction_id]