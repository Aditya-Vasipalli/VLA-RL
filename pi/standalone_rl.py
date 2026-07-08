#!/usr/bin/env python3
"""
STANDALONE Hierarchical RL Controller
No external dependencies from EDAI folder - completely self-contained!

Uses:
- hardware.py (TB6612 + ultrasonics) 
- YOLO server at http://10.168.71.19:8000 for vision
- Simple Q-table for navigation
"""

import time
import threading
import logging
import json
import requests
import cv2
import numpy as np
import configparser
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from hardware import RobotHardware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
config = configparser.ConfigParser()
config.read('config.ini')
SERVER_URL = config.get('server', 'url', fallback='http://10.168.71.19:8000')
TARGET_CLASS = config.get('target', 'object_class', fallback='bottle')
ALPHA = config.getfloat('rl', 'alpha', fallback=0.1)
GAMMA = config.getfloat('rl', 'gamma', fallback=0.95)
EPSILON = config.getfloat('rl', 'epsilon', fallback=0.1)

# =================================================================
# STATE DEFINITIONS (Simplified)
# =================================================================

class RobotState(Enum):
    """High-level states"""
    TARGET_VISIBLE = "TARGET_VISIBLE"
    SEARCH_ROTATE = "SEARCH_ROTATE"
    EXPLORATION = "EXPLORATION"
    OBSTACLE_AVOID = "OBSTACLE_AVOID"

class Visibility(Enum):
    NOT_VISIBLE = 0
    VISIBLE = 1

class Position(Enum):
    LEFT = "L"
    CENTER = "C"
    RIGHT = "R"
    UNKNOWN = "X"

class Distance(Enum):
    NEAR = "NEAR"    # < 0.5m
    MID = "MID"      # 0.5-1.5m
    FAR = "FAR"      # > 1.5m
    UNKNOWN = "X"

class Action(Enum):
    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

# =================================================================
# YOLO CLIENT (Talks to Server)
# =================================================================

class YOLOClient:
    """Simple client for YOLO server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"YOLO Client initialized: {server_url}")
    
    def detect(self, target_class: str) -> Optional[Dict]:
        """
        Get detection from YOLO server
        Returns: {'confidence': float, 'bbox': [x,y,w,h], 'distance': float} or None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/detect",
                files={'image': buffer.tobytes()},
                data={'confidence': 0.5},
                timeout=2.0
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            detections = data.get('detections', [])
            
            # Find target class
            for det in detections:
                if det['class'].lower() == target_class.lower():
                    # Estimate distance from bbox area
                    bbox = det['bbox']
                    area = bbox[2] * bbox[3]
                    
                    # Simple distance estimation
                    if area > 50000:
                        distance = 0.3
                    elif area > 20000:
                        distance = 0.8
                    elif area > 5000:
                        distance = 1.5
                    else:
                        distance = 3.0
                    
                    return {
                        'confidence': det['confidence'],
                        'bbox': bbox,
                        'distance': distance,
                        'center_x': bbox[0] + bbox[2] // 2,
                        'center_y': bbox[1] + bbox[3] // 2
                    }
            
            return None
        
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return None
    
    def release(self):
        self.cap.release()

# =================================================================
# SIMPLE Q-LEARNING
# =================================================================

class SimpleQLearning:
    """Minimal Q-learning for navigation"""
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        logger.info("Q-Learning initialized")
    
    def get_state(self, detection: Optional[Dict], distances: Dict) -> Tuple:
        """Convert sensor readings to state tuple"""
        if detection and detection['confidence'] > 0.5:
            vis = 1
            
            # Position (left/center/right)
            center_x = detection['center_x']
            if center_x < 220:
                pos = 'L'
            elif center_x > 420:
                pos = 'R'
            else:
                pos = 'C'
            
            # Distance
            dist = detection['distance']
            if dist < 0.5:
                dist_bin = 'NEAR'
            elif dist < 1.5:
                dist_bin = 'MID'
            else:
                dist_bin = 'FAR'
        else:
            vis = 0
            pos = 'X'
            dist_bin = 'X'
        
        # Obstacle
        front = distances.get('front', 10)
        if front and front < 0.3:
            obstacle = 'NEAR'
        else:
            obstacle = 'CLEAR'
        
        return (vis, pos, dist_bin, obstacle)
    
    def choose_action(self, state: Tuple) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Q-learning update"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def save(self, filename='q_table.json'):
        """Save Q-table"""
        data = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(data, f)
        logger.info(f"Q-table saved: {len(self.q_table)} states")
    
    def load(self, filename='q_table.json'):
        """Load Q-table"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                state = eval(k)  # Convert string back to tuple
                self.q_table[state] = np.array(v)
            logger.info(f"Q-table loaded: {len(self.q_table)} states")
        except FileNotFoundError:
            logger.info("No existing Q-table found, starting fresh")

# =================================================================
# REWARD CALCULATOR (Your VLA System)
# =================================================================

class RewardCalculator:
    """Calculate rewards based on VLA system"""
    
    def __init__(self):
        self.last_detection = None
        self.last_centered = False
        self.last_distance = None
    
    def calculate(self, detection: Optional[Dict], distances: Dict) -> float:
        """
        Calculate reward preserving VLA logic:
        - Target in frame: +5 (0→1 transition)
        - Target centered: +2 (±50px margin)
        - Distance improved: +1
        - Target lost: -2
        - Collision risk: -50
        - Goal reached: +100
        """
        reward = -0.1  # Step penalty
        
        current_visible = detection is not None and detection['confidence'] > 0.5
        last_visible = self.last_detection is not None
        
        if current_visible:
            # TARGET IN FRAME
            if not last_visible:
                reward += 5  # 0→1 transition
                logger.info("TARGET ACQUIRED: +5")
            else:
                reward += 0.5  # Maintain visibility
            
            # TARGET CENTERED (Y-axis, ±50px margin)
            center_y = detection['center_y']
            y_error = abs(center_y - 240)
            
            if y_error < 50:
                if not self.last_centered:
                    reward += 2
                    logger.info("TARGET CENTERED: +2")
                else:
                    reward += 1
                self.last_centered = True
            else:
                if self.last_centered:
                    reward -= 0.5
                self.last_centered = False
            
            # DISTANCE REWARD
            current_dist = detection['distance']
            if self.last_distance:
                if current_dist < self.last_distance:
                    reward += 1  # Getting closer
                elif current_dist > self.last_distance:
                    reward -= 0.2  # Moving away
            
            self.last_distance = current_dist
            
            # GOAL REACHED
            if current_dist < 0.35:
                reward += 100
                logger.info("GOAL REACHED: +100")
        
        else:
            # TARGET LOST
            if last_visible:
                reward -= 2
                logger.warning("TARGET LOST: -2")
            self.last_centered = False
            self.last_distance = None
        
        # COLLISION RISK
        front = distances.get('front', 10)
        if front and front < 0.15:
            reward -= 50
            logger.warning("COLLISION IMMINENT: -50")
        elif front and front < 0.25:
            reward -= 2
        
        self.last_detection = detection
        return reward

# =================================================================
# LAYER 1: SAFETY
# =================================================================

class SafetyLayer:
    """Fast safety checks using ultrasonics only"""
    
    def __init__(self, hardware: RobotHardware):
        self.hardware = hardware
        self.COLLISION_THRESH = 0.15
        self.SAFETY_THRESH = 0.20
        logger.info("Safety Layer initialized")
    
    def check_safe(self, action: str, distances: Dict) -> Tuple[bool, str]:
        """Check if action is safe"""
        front = distances.get('front', 10) or 10
        left = distances.get('left', 10) or 10
        right = distances.get('right', 10) or 10
        
        if action in ['forward', 'FORWARD'] and front < self.COLLISION_THRESH:
            return False, "FRONT COLLISION"
        
        if front < self.COLLISION_THRESH:
            return False, "FRONT BLOCKED"
        
        if left < 0.12 and action in ['turn_left', 'TURN_LEFT']:
            return False, "LEFT BLOCKED"
        
        if right < 0.12 and action in ['turn_right', 'TURN_RIGHT']:
            return False, "RIGHT BLOCKED"
        
        return True, "OK"
    
    def get_safe_action(self, distances: Dict) -> str:
        """Emergency safe action"""
        back = distances.get('back', 10) or 10
        left = distances.get('left', 10) or 10
        right = distances.get('right', 10) or 10
        
        if back > 0.3:
            return "backward"
        elif left > right:
            return "turn_left"
        else:
            return "turn_right"

# =================================================================
# LAYER 2: EXPLORATION
# =================================================================

class ExplorationLayer:
    """Wall-following and exploration"""
    
    def __init__(self):
        self.wall_follow_distance = 0.30
        self.stuck_counter = 0
        logger.info("Exploration Layer initialized")
    
    def explore(self, distances: Dict) -> str:
        """Decide exploration action"""
        front = distances.get('front', 10) or 10
        left = distances.get('left', 10) or 10
        
        # Simple wall-following
        if front < 0.4:
            return "turn_right"
        
        if left < self.wall_follow_distance - 0.1:
            return "turn_right"
        elif left > self.wall_follow_distance + 0.1:
            return "turn_left"
        else:
            return "forward"

# =================================================================
# MAIN CONTROLLER
# =================================================================

class HierarchicalController:
    """Main hierarchical controller - STANDALONE VERSION"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("STANDALONE HIERARCHICAL RL CONTROLLER")
        logger.info("=" * 60)
        
        # Initialize components
        self.hardware = RobotHardware()
        self.yolo = YOLOClient(SERVER_URL)
        self.qlearning = SimpleQLearning(ALPHA, GAMMA, EPSILON)
        self.qlearning.load()  # Load existing Q-table if available
        self.rewards = RewardCalculator()
        
        # Initialize layers
        self.safety = SafetyLayer(self.hardware)
        self.exploration = ExplorationLayer()
        
        # State machine
        self.state = RobotState.SEARCH_ROTATE
        self.running = False
        self.search_counter = 0
        
        # RL state
        self.current_state = None
        self.last_action = None
        
        logger.info("Initialization complete")
        logger.info("=" * 60)
    
    def start(self):
        """Start control loop"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        logger.info("Control loop started")
    
    def stop(self):
        """Stop control loop"""
        self.running = False
        self.hardware.stop()
        self.yolo.release()
        self.qlearning.save()
        logger.info("Stopped")
    
    def _control_loop(self):
        """Main control loop"""
        while self.running:
            try:
                # Get sensors
                distances = self.hardware.get_all_distances()
                detection = self.yolo.detect(TARGET_CLASS)
                
                # Decide action based on state
                if self.state == RobotState.TARGET_VISIBLE:
                    action = self._handle_target_visible(detection, distances)
                
                elif self.state == RobotState.SEARCH_ROTATE:
                    action = self._handle_search(detection, distances)
                
                elif self.state == RobotState.EXPLORATION:
                    action = self._handle_exploration(detection, distances)
                
                elif self.state == RobotState.OBSTACLE_AVOID:
                    action = self._handle_obstacle(distances)
                
                else:
                    action = "stop"
                
                # Safety check
                is_safe, reason = self.safety.check_safe(action, distances)
                if not is_safe:
                    logger.warning(f"SAFETY OVERRIDE: {reason}")
                    self.state = RobotState.OBSTACLE_AVOID
                    action = self.safety.get_safe_action(distances)
                
                # Execute
                self._execute(action)
                
                time.sleep(0.5)  # 2Hz main loop
            
            except Exception as e:
                logger.error(f"Control loop error: {e}", exc_info=True)
                self.hardware.stop()
                time.sleep(1)
    
    def _handle_target_visible(self, detection, distances):
        """Layer 3: Q-learning goal seeking"""
        if not detection:
            self.state = RobotState.SEARCH_ROTATE
            self.search_counter = 0
            return "stop"
        
        # Get Q-learning state
        state = self.qlearning.get_state(detection, distances)
        
        # Calculate reward
        if self.current_state and self.last_action is not None:
            reward = self.rewards.calculate(detection, distances)
            self.qlearning.update(self.current_state, self.last_action, reward, state)
        
        # Choose action
        action_idx = self.qlearning.choose_action(state)
        action_map = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        action = action_map[action_idx]
        
        # Update state
        self.current_state = state
        self.last_action = action_idx
        
        return action
    
    def _handle_search(self, detection, distances):
        """Search by rotating"""
        if detection:
            self.state = RobotState.TARGET_VISIBLE
            logger.info("Target reacquired!")
            return "stop"
        
        self.search_counter += 1
        if self.search_counter > 12:  # 360°
            self.state = RobotState.EXPLORATION
            self.search_counter = 0
            logger.info("Target not found, exploring...")
        
        return "turn_right"
    
    def _handle_exploration(self, detection, distances):
        """Layer 2: Wall-following"""
        if detection:
            self.state = RobotState.TARGET_VISIBLE
            logger.info("Target found during exploration!")
            return "stop"
        
        return self.exploration.explore(distances)
    
    def _handle_obstacle(self, distances):
        """Emergency obstacle avoidance"""
        action = self.safety.get_safe_action(distances)
        
        # Check if cleared
        front = distances.get('front', 10) or 10
        if front > 0.3:
            self.state = RobotState.SEARCH_ROTATE
            logger.info("Obstacle cleared")
        
        return action
    
    def _execute(self, action: str):
        """Execute motor command"""
        if action == "stop":
            self.hardware.stop()
        elif action == "forward":
            self.hardware.move_forward(speed=50, duration=0.4)
        elif action == "backward":
            self.hardware.move_backward(speed=50, duration=0.4)
        elif action == "turn_left":
            self.hardware.turn_left(speed=40, duration=0.3)
        elif action == "turn_right":
            self.hardware.turn_right(speed=40, duration=0.3)

# =================================================================
# MAIN
# =================================================================

def main():
    controller = HierarchicalController()
    
    try:
        controller.start()
        logger.info("Running... Press Ctrl+C to stop")
        
        while controller.running:
            time.sleep(5)
            logger.info(f"State: {controller.state.value}, "
                       f"Q-table: {len(controller.qlearning.q_table)} states")
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()
