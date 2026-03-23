#!/usr/bin/env python3
"""
VLA SERVER - Vision-Language-Action Processing Server

Optimized VLA server with web interface for monitoring robot missions.
Uses LLaVA 7B model with sub-5s response time for real-time object detection.

Usage:
    python vla_server.py
    
Web interface: http://localhost:5000
VLA API: TCP socket on port 9999
"""

import socket
import struct
import json
import numpy as np
import threading
import time
import base64
import requests
import cv2
import re
from typing import Dict, Optional, Tuple
from flask import Flask, Response, render_template_string, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================
# CONFIGURATION
# =================================================================

HOST = '0.0.0.0'  # Listen on all interfaces
VLA_PORT = 9999   # TCP socket port
WEB_PORT = 5000   # Web interface port
OLLAMA_URL = "http://localhost:11434/api/generate"

# Optimization settings for llava:7b (balanced for quality)
OPTIMAL_IMAGE_SIZE = (640, 480)  # Higher resolution for better clarity
OPTIMAL_JPEG_QUALITY = 85        # Much higher quality for better detail
LLAVA_MODEL = "llava:7b"         # Switch to llava:7b
OPTIMAL_PROMPT = (
    "Describe what you see in this image. Is there a {target} visible? If yes, where is it located?\n\n"
    "Answer in this exact format:\n"
    "Position: left\n"
    "Distance: near\n" 
    "Confidence: 90%\n"
    "Bearing: -0.5\n\n"
    "Position can be: left, center, right, or unknown\n"
    "Distance can be: near, far, or unknown\n"
    "Confidence: 0-100% how certain you are\n"
    "Bearing: -1.0 (far left) to 1.0 (far right), 0.0 = center"
)

# =================================================================
# VLA PROCESSING
# =================================================================

class VLAProcessor:
    """Optimized VLA processing with NLP parsing"""
    
    def __init__(self, model_name=LLAVA_MODEL):
        self.model_name = model_name
        self.current_target = "bottle"  # Default target
        self.sensor_data = {'F': 0, 'L': 0, 'R': 0, 'B': 0}  # Ultrasonic sensors
        self.rl_stats = {'episode': 0, 'mode': 'exploration', 'q_table_size': 0, 'epsilon': 0.1}
        self.stats = {
            'total_requests': 0,
            'avg_response_time': 0,
            'last_response_time': 0,
            'successful_detections': 0
        }
    
    def optimize_image(self, image_bytes: bytes) -> str:
        """Smart image preprocessing for sub-4s VLA processing"""
        try:
            # Decode image from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return base64.b64encode(image_bytes).decode('utf-8')
            
            # 1. Smart crop - remove unnecessary border areas (10% margin)
            h, w = image.shape[:2]
            crop_margin = min(h, w) // 10
            if crop_margin > 0:
                image = image[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
            
            # 2. Enhance contrast for better VLA object recognition
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Optimal resize for VLA clarity (larger for better object recognition)
            target_size = (480, 360)  # Much larger for better detail
            h, w = image.shape[:2]
            
            # Maintain aspect ratio
            if h > w:
                new_h = target_size[1]
                new_w = int(w * (target_size[1] / h))
            else:
                new_w = target_size[0]
                new_h = int(h * (target_size[0] / w))
            
            # Resize with good quality but fast interpolation
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 4. Pad to exact target size (maintains aspect ratio)
            if new_h < target_size[1] or new_w < target_size[0]:
                canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                y_offset = (target_size[1] - new_h) // 2
                x_offset = (target_size[0] - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                resized = canvas
            
            # 5. VLA-specific optimizations
            # Boost saturation for better object distinction
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.1)  # 10% more saturation
            resized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Light denoising for better compression
            resized = cv2.medianBlur(resized, 3)
            
            # 6. High-quality JPEG compression for clarity
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 85,  # Higher quality for clarity
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            success, buffer = cv2.imencode('.jpg', resized, encode_params)
            
            if success:
                return base64.b64encode(buffer).decode('utf-8')
            else:
                # Fallback to original
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            # Fallback to original image
            return base64.b64encode(image_bytes).decode('utf-8')
    
    def set_target(self, new_target: str):
        """Change the target object"""
        self.current_target = new_target.lower().strip()
        logger.info(f"🎯 Target changed to: {self.current_target}")
    
    def update_sensor_data(self, sensors: Dict):
        """Update ultrasonic sensor readings"""
        self.sensor_data = sensors
    
    def update_rl_stats(self, rl_data: Dict):
        """Update RL agent statistics"""
        self.rl_stats.update(rl_data)
    
    def set_target(self, new_target: str):
        """Change the target object"""
        self.current_target = new_target.lower().strip()
        logger.info(f"🎯 Target changed to: {self.current_target}")
    
    def update_sensor_data(self, sensors: Dict):
        """Update ultrasonic sensor readings"""
        self.sensor_data = sensors
    
    def update_rl_stats(self, rl_data: Dict):
        """Update RL agent statistics"""
        self.rl_stats.update(rl_data)
    
    def parse_response_nlp(self, response: str, target: str = "bottle") -> Dict:
        """Parse LLaVA response - handles both structured and natural language responses."""
        import re
        response_stripped = response.strip()
        
        # Check for explicit negative (target-specific)
        if re.search(rf'no (?:{re.escape(target)}|[\w ]+)? visible', response_stripped, re.IGNORECASE):
            return {
                'target_found': False,
                'position': 'unknown',
                'distance': 'unknown',
                'confidence': 0.05,
                'bearing': 0.0,
                'raw_response': response_stripped[:100] + "..." if len(response_stripped) > 100 else response_stripped
            }

        # Extract fields using regex (structured format)
        def extract_field(field, default):
            match = re.search(rf'{field}\s*:\s*([^\n]+)', response_stripped, re.IGNORECASE)
            return match.group(1).strip() if match else default

        position = extract_field('Position', 'unknown').lower()
        distance = extract_field('Distance', 'unknown').lower()
        confidence_str = extract_field('Confidence', '70%').replace('%','').strip()
        bearing_str = extract_field('Bearing', 'auto').lower()

        # If structured format not found, try to parse natural language
        if position == 'unknown' and target.lower() in response_stripped.lower():
            # Look for position indicators in natural language
            if re.search(r'\b(left|to the left|on the left)\b', response_stripped, re.IGNORECASE):
                position = 'left'
            elif re.search(r'\b(right|to the right|on the right)\b', response_stripped, re.IGNORECASE):
                position = 'right'
            elif re.search(r'\b(center|centre|middle|central)\b', response_stripped, re.IGNORECASE):
                position = 'center'
            else:
                position = 'center'  # Default if target detected but no position specified
            
            # Look for distance indicators
            if re.search(r'\b(close|near|front|foreground)\b', response_stripped, re.IGNORECASE):
                distance = 'near'
            elif re.search(r'\b(far|distant|back|background)\b', response_stripped, re.IGNORECASE):
                distance = 'far'
            else:
                distance = 'near'  # Default assumption

        # Parse confidence as float (0-1)
        try:
            confidence = float(confidence_str) / 100.0
        except Exception:
            # If target detected but no confidence, assume high confidence
            confidence = 0.85 if target.lower() in response_stripped.lower() else 0.05
        confidence = min(max(confidence, 0.05), 0.99)

        # Parse bearing - try as float first, then map from position
        try:
            bearing = float(bearing_str)
            bearing = min(max(bearing, -1.0), 1.0)  # Clamp to [-1, 1]
        except:
            # Map from position if bearing not provided
            if position == 'left':
                bearing = -0.7
            elif position == 'right':
                bearing = 0.7
            elif position == 'center':
                bearing = 0.0
            else:
                bearing = 0.0

        # Determine if target is found
        target_found = (
            position in ['left', 'center', 'right'] or
            distance in ['near', 'far'] or
            confidence > 0.2 or
            target.lower() in response_stripped.lower()
        )

        return {
            'target_found': target_found,
            'position': position,
            'distance': distance,
            'confidence': confidence,
            'bearing': bearing,
            'raw_response': response_stripped[:100] + "..." if len(response_stripped) > 100 else response_stripped
        }
    
    def _position_to_bearing(self, position: str) -> float:
        """Convert position to bearing (-1.0 to 1.0)"""
        if position == 'left':
            return -0.7
        elif position == 'right':
            return 0.7
        elif position == 'center':
            return 0.0
        else:
            return 0.0  # Unknown position
    
    def process_image(self, image_bytes: bytes, target: str = None) -> Dict:
        """Process image and return VLA result"""
        start_time = time.time()
        
        try:
            # Use current target if not specified
            if target is None:
                target = self.current_target
            
            # Optimize image
            image_b64 = self.optimize_image(image_bytes)
            
            # Create target-specific prompt
            target_prompt = OPTIMAL_PROMPT.format(target=target)
            
            # Prepare VLA request (optimized for llava:7b)
            payload = {
                'model': self.model_name,
                'prompt': target_prompt,
                'images': [image_b64],
                'stream': False,
                'options': {
                    'temperature': 0.1,    # Slightly higher for better reasoning
                    'top_k': 10,           # More options for better detection
                    'top_p': 0.3,          # Higher for more detailed responses
                    'num_ctx': 512,        # More context for better understanding
                    'num_predict': 100,    # Longer responses for complete format
                    'num_thread': 8,       # Use all laptop cores
                    'repeat_penalty': 1.1, # Slight penalty to avoid repetition
                    'stop': [],            # Don't stop early
                }
            }
            
            # Call VLA model with aggressive timeout for speed
            response = requests.post(OLLAMA_URL, json=payload, timeout=8)
            
            if response.status_code == 200:
                result = response.json()
                vla_response = result.get('response', '').strip()
                
                # Parse with NLP
                parsed = self.parse_response_nlp(vla_response, target)
                
                # Update stats
                response_time = time.time() - start_time
                self.stats['total_requests'] += 1
                self.stats['last_response_time'] = response_time
                self.stats['avg_response_time'] = (
                    self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + response_time
                ) / self.stats['total_requests']
                
                if parsed['target_found']:
                    self.stats['successful_detections'] += 1
                
                logger.info(f"VLA processed in {response_time:.2f}s: {parsed['target_found']} at {parsed['position']}")
                
                return {
                    'success': True,
                    'response_time': response_time,
                    'target': target,
                    **parsed
                }
            else:
                logger.error(f"VLA request failed: HTTP {response.status_code}")
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"VLA processing error: {e}")
            return {'success': False, 'error': str(e)}

# =================================================================
# TCP SERVER FOR RPI COMMUNICATION
# =================================================================

class VLATCPServer:
    """TCP server for Pi communication"""
    
    def __init__(self, vla_processor: VLAProcessor):
        self.vla_processor = vla_processor
        self.latest_image = None
        self.latest_result = None
        self.clients = []
    
    def start_server(self):
        """Start TCP server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, VLA_PORT))
        server_socket.listen(5)
        
        logger.info(f"VLA TCP server listening on {HOST}:{VLA_PORT}")
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                logger.info(f"Client connected: {address}")
                
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                logger.error(f"Server error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle Pi client connection"""
        try:
            while True:
                # Receive message length
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                message_length = struct.unpack('!I', length_data)[0]
                
                # Receive message data
                message_data = b''
                while len(message_data) < message_length:
                    chunk = client_socket.recv(min(message_length - len(message_data), 4096))
                    if not chunk:
                        break
                    message_data += chunk
                
                # Parse request
                request = json.loads(message_data.decode('utf-8'))
                
                if request['type'] == 'vla_query':
                    # Process image with VLA
                    image_bytes = base64.b64decode(request['image'])
                    target = request.get('target', self.vla_processor.current_target)
                    
                    # Store for web interface
                    self.latest_image = image_bytes
                    
                    # Process with VLA
                    result = self.vla_processor.process_image(image_bytes, target)
                    self.latest_result = result
                    
                    # Send response
                    response = json.dumps(result).encode('utf-8')
                    client_socket.send(struct.pack('!I', len(response)))
                    client_socket.send(response)
                    
                    logger.info(f"Sent VLA result to {address}: {result.get('target_found', False)}")
                
                elif request['type'] == 'sensor_update':
                    # Update sensor data
                    self.vla_processor.update_sensor_data(request.get('sensors', {}))
                    
                elif request['type'] == 'rl_update':
                    # Update RL statistics
                    self.vla_processor.update_rl_stats(request.get('rl_data', {}))
                
                elif request['type'] == 'target_change':
                    # Change target
                    new_target = request.get('target', 'bottle')
                    self.vla_processor.set_target(new_target)
                    
                    # Send confirmation
                    response_data = {'success': True, 'new_target': self.vla_processor.current_target}
                    response = json.dumps(response_data).encode('utf-8')
                    client_socket.send(struct.pack('!I', len(response)))
                    client_socket.send(response)
                
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
            logger.info(f"Client disconnected: {address}")

# =================================================================
# WEB INTERFACE
# =================================================================

# Global variables for web interface
vla_processor = None
tcp_server = None

app = Flask(__name__)

WEB_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VLA+RL Robot Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .stat-label { color: #666; margin-top: 5px; }
        .image-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .vla-result { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .success { color: #4CAF50; }
        .failed { color: #f44336; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; }
        .status.found { background: #4CAF50; }
        .status.not-found { background: #f44336; }
        img { max-width: 100%; height: auto; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VLA+RL Robot Monitor</h1>
            <p>Real-time monitoring of Vision-Language-Action system</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-requests">-</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-time">-</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="last-time">-</div>
                <div class="stat-label">Last Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="detection-rate">-</div>
                <div class="stat-label">Detection Rate</div>
            </div>
        </div>
        
        <div class="image-section">
            <h3>Latest Camera Feed</h3>
            <img id="latest-image" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="No image received yet">
        </div>
        
        <div class="vla-result">
            <h3>Latest VLA Analysis</h3>
            <div id="vla-content">
                <p>Waiting for VLA analysis...</p>
            </div>
        </div>
    </div>
    
    <script>
        function updateData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update stats
                    document.getElementById('total-requests').textContent = data.stats.total_requests;
                    document.getElementById('avg-time').textContent = data.stats.avg_response_time.toFixed(2) + 's';
                    document.getElementById('last-time').textContent = data.stats.last_response_time.toFixed(2) + 's';
                    
                    const detectionRate = data.stats.total_requests > 0 ? 
                        (data.stats.successful_detections / data.stats.total_requests * 100).toFixed(1) : 0;
                    document.getElementById('detection-rate').textContent = detectionRate + '%';
                    
                    // Update image
                    if (data.latest_image) {
                        document.getElementById('latest-image').src = 'data:image/jpeg;base64,' + data.latest_image;
                    }
                    
                    // Update VLA result
                    if (data.latest_result) {
                        const result = data.latest_result;
                        const statusClass = result.target_found ? 'found' : 'not-found';
                        const statusText = result.target_found ? 'FOUND' : 'NOT FOUND';
                        
                        document.getElementById('vla-content').innerHTML = `
                            <div style="margin-bottom: 15px;">
                                <span class="status ${statusClass}">${statusText}</span>
                            </div>
                            <p><strong>Position:</strong> ${result.position}</p>
                            <p><strong>Distance:</strong> ${result.distance}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Bearing:</strong> ${result.bearing.toFixed(2)}</p>
                            <p><strong>Response Time:</strong> ${result.response_time.toFixed(2)}s</p>
                            <p><strong>Raw Response:</strong> ${result.raw_response}</p>
                        `;
                    }
                })
                .catch(error => console.error('Error updating data:', error));
        }
        
        // Update every 2 seconds
        setInterval(updateData, 2000);
        updateData(); // Initial load
    </script>
</body>
</html>
"""

@app.route('/')
def index():
        # Add a simple form to set the target
        global vla_processor
        return render_template_string(WEB_TEMPLATE + """
        <hr>
        <form id='targetForm' method='post' action='/api/target' onsubmit='event.preventDefault(); setTarget();'>
            <label for='target'>Target object:</label>
            <input type='text' id='target' name='target' value='{{ current_target }}' />
            <button type='submit'>Set Target</button>
        </form>
        <script>
        function setTarget() {
            const target = document.getElementById('target').value;
            fetch('/api/target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target })
            })
            .then(r => r.json())
            .then(data => { if(data.success){ alert('Target set to: ' + data.new_target); } else { alert('Error: ' + data.error); } });
        }
        </script>
        """, current_target=vla_processor.current_target)

@app.route('/api/status')
def api_status():
    """API endpoint for current status"""
    global vla_processor, tcp_server
    
    response = {
        'stats': vla_processor.stats if vla_processor else {},
        'latest_image': None,
        'latest_result': None
    }
    
    if tcp_server:
        if tcp_server.latest_image:
            response['latest_image'] = base64.b64encode(tcp_server.latest_image).decode('utf-8')
        response['latest_result'] = tcp_server.latest_result
    
    return jsonify(response)

@app.route('/api/latest')
def api_latest():
    """API endpoint for latest VLA result with sensor and RL data"""
    global vla_processor, tcp_server
    
    if tcp_server and tcp_server.latest_result:
        return jsonify({
            'result': tcp_server.latest_result,
            'sensors': vla_processor.sensor_data,
            'rl_stats': vla_processor.rl_stats,
            'current_target': vla_processor.current_target,
            'timestamp': time.time()
        })
    return jsonify({
        'result': None, 
        'sensors': vla_processor.sensor_data,
        'rl_stats': vla_processor.rl_stats,
        'current_target': vla_processor.current_target,
        'timestamp': time.time()
    })

@app.route('/api/target', methods=['POST'])
def change_target():
    """API endpoint to change target"""
    data = request.get_json()
    new_target = data.get('target', '').strip()
    
    if new_target:
        vla_processor.set_target(new_target)
        return jsonify({'success': True, 'new_target': vla_processor.current_target})
    else:
        return jsonify({'success': False, 'error': 'Invalid target'}), 400

# =================================================================
# MAIN SERVER
# =================================================================

def main():
    """Start VLA server with web interface"""
    global vla_processor, tcp_server
    
    print("STARTING VLA SERVER")
    print("=" * 50)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        print("Ollama is running")
    except:
        print("Ollama not running. Please start it first:")
        print("   ollama serve")
        return
    
    # Initialize VLA processor
    vla_processor = VLAProcessor()
    print("VLA processor initialized")
    
    # Initialize TCP server
    tcp_server = VLATCPServer(vla_processor)
    
    # Start TCP server in background thread
    tcp_thread = threading.Thread(target=tcp_server.start_server)
    tcp_thread.daemon = True
    tcp_thread.start()
    
    print(f"TCP server started on port {VLA_PORT}")
    print(f"Web interface starting on http://localhost:{WEB_PORT}")
    print("=" * 50)
    print("Ready for robot connections!")
    
    # Start web interface (this blocks)
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False)

if __name__ == "__main__":
    main()