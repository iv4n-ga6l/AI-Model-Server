# AI Model Server

A high-performance web server for AI models, primarily focused on computer vision tasks. This server provides a simple and efficient way to deploy and use AI models through a RESTful API.

## Architecture

```
--------------------------------------------------------- 
                  AI Model Server 
--------------------------------------------------------- 

---------------------- Model Service --------------------- 
|  Web Service  |  Task Schedule  |  Service Manage      | 
|                        WORKFLOW                        | 
--------------------------------------------------------- 

---------------------- Model Deploy ---------------------- 
| Model Deploy | GPU Backend | CPU Backend | Model Quant | 
|                        MNN                             | 
--------------------------------------------------------- 

----------------------- Model Dev ------------------------ 
| Classification | Segmentation | Detection | OCR | etc. | 
|       TensorFlow       ↑        PyTorch                | 
---------------------------------------------------------
```

## Features

- **FastAPI Backend**: High-performance asynchronous API
- **YOLOv8 Integration**: Default model for object detection and segmentation
- **Multiple Task Support**: Detection, segmentation, and more
- **Asynchronous Processing**: Handle multiple requests efficiently
- **Result Visualization**: View detection and segmentation results
- **Model Management**: Load, unload, and switch between models

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository

```bash
git clone https://github.com/iv4n-ga6l/AI-Model-Server.git
cd AI-Model-Server
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the server

```bash
uvicorn main:app --reload
```

The server will be available at http://localhost:8000

## API Endpoints

### GET /

Root endpoint, returns a welcome message and server status.

### GET /models

List all available models.

### POST /predict

Run inference on an uploaded image.

**Parameters:**
- `file`: The image file to analyze (required)
- `model_name`: Name of the model to use (default: "yolov8n")
- `task_type`: Type of task ("detection" or "segmentation", default: "detection")
- `confidence`: Confidence threshold (default: 0.5)

### GET /status/{prediction_id}

Get the status of a prediction.

### GET /result/{prediction_id}

Get the result image of a prediction.

### GET /health

Health check endpoint.

## Models

The server comes with the following pre-configured YOLOv8 models:

- `yolov8n`: YOLOv8 Nano (detection)
- `yolov8s`: YOLOv8 Small (detection)
- `yolov8m`: YOLOv8 Medium (detection)
- `yolov8n-seg`: YOLOv8 Nano (segmentation)
- `yolov8s-seg`: YOLOv8 Small (segmentation)

## Project Structure

```
ai-model-server/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── inference_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py
│   └── __init__.py
├── static/
│   ├── uploads/
│   └── results/
├── models/
├── main.py
└── requirements.txt
```

## License

MIT

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)