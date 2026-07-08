#!/usr/bin/env python3
"""
Minimal YOLO Inference Server
Runs on Windows laptop with GPU for robot object detection

Features:
- GPU-accelerated YOLOv8 inference
- Flask REST API for Pi communication
- Health check endpoint
- Simple and efficient
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model = None
device = None
inference_count = 0
total_inference_time = 0


def setup_gpu():
    """Setup GPU device"""
    global device
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = 'cpu'
        logger.warning("⚠️ No GPU available, using CPU (slower)")
    
    return device


def load_model(model_path='yolov8s.pt'):
    """Load YOLO model and warm up"""
    global model
    
    logger.info(f"📦 Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    # Warm up model with dummy inference
    logger.info("🔥 Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        with torch.no_grad():
            model(dummy, verbose=False)
    
    logger.info("✅ Model ready for inference")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'model_loaded': model is not None,
        'inference_count': inference_count,
        'avg_inference_time': f"{total_inference_time / max(inference_count, 1) * 1000:.1f}ms"
    })


@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Object detection endpoint
    
    Request JSON:
        {
            "image": "hex_encoded_jpeg_bytes"
        }
    
    Response JSON:
        {
            "detections": [
                {
                    "class": "bottle",
                    "confidence": 0.85,
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ],
            "inference_time_ms": 48.5
        }
    """
    global inference_count, total_inference_time
    
    try:
        # Get image from request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode hex image data
        img_hex = data['image']
        img_bytes = bytes.fromhex(img_hex)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run YOLO inference
        start_time = time.time()
        with torch.no_grad():
            results = model(frame, verbose=False)[0]
        inference_time = time.time() - start_time
        
        # Update statistics
        inference_count += 1
        total_inference_time += inference_time
        
        # Parse results
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            
            # Only include high-confidence detections
            if confidence > 0.5:
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        logger.info(f"🔍 Detected {len(detections)} objects in {inference_time*1000:.1f}ms")
        
        return jsonify({
            'detections': detections,
            'inference_time_ms': inference_time * 1000,
            'frame_size': frame.shape[:2]
        })
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Simple status page"""
    return """
    <html>
    <head><title>YOLO Server</title></head>
    <body style="font-family: Arial; padding: 20px; background: #1a1a2e; color: #eee;">
        <h1>🤖 YOLO Inference Server</h1>
        <p>Status: <strong style="color: #4CAF50;">ONLINE</strong></p>
        <p>GPU: <strong>{}</strong></p>
        <p>Inferences: <strong>{}</strong></p>
        <p>Avg Time: <strong>{:.1f}ms</strong></p>
        <hr>
        <h3>API Endpoints:</h3>
        <ul>
            <li><code>GET /health</code> - Health check</li>
            <li><code>POST /detect</code> - Object detection</li>
        </ul>
    </body>
    </html>
    """.format(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        inference_count,
        (total_inference_time / max(inference_count, 1)) * 1000
    )


def main():
    """Main entry point"""
    logger.info("🖥️  Starting YOLO Inference Server...")
    
    # Setup GPU
    setup_gpu()
    
    # Load model (use absolute path to avoid confusion)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yolov8s.pt')
    load_model(model_path)
    
    # Start Flask server
    logger.info("🌐 Starting Flask server on http://0.0.0.0:8000")
    logger.info("📡 Access from Pi using your laptop IP address")
    logger.info("   Example: http://192.168.1.100:8000")
    
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=8000,
        debug=False,
        threaded=True  # Handle multiple requests
    )


if __name__ == '__main__':
    main()
