#!/usr/bin/env python3
"""
Minimal RL Robot Controller with Q-Learning + YOLO
Runs on Raspberry Pi 4B with continuous learning

Architecture:
- Pi: Q-learning, sensors, motors, camera capture
- Server: GPU YOLO inference (with fallback to color detection)
"""

import numpy as np
import cv2
import requests
import time
import json
import logging
import configparser
from collections import deque
from typing import Dict, Tuple, Optional
from hardware import RobotHardware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinimalRLRobot:
    """Q-Learning robot with hybrid YOLO/color detection"""
    
    def __init__(self, config_file='config.ini'):
        logger.info("🤖 Initializing RL Robot...")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Hardware
        self.robot = RobotHardware()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Q-Learning parameters
        self.alpha = self.config['rl']['learning_rate']  # Learning rate
        self.gamma = self.config['rl']['discount_factor']  # Discount factor
        self.epsilon = self.config['rl']['epsilon']  # Exploration rate
        
        # State/Action spaces
        self.num_states = 96  # 2(vis) × 4(pos) × 3(dist) × 4(ultrasonic)
        self.num_actions = 4  # stop, forward, left, right
        self.action_names = ['stop', 'forward', 'left', 'right']
        
        # Q-table (state -> action values)
        self.Q = self._load_or_create_qtable()
        
        # Server configuration
        self.server_url = self.config['server']['url']
        self.server_available = False
        self.target_object = self.config['target']['object_class']
        
        # Episode tracking
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Test server connection
        self._test_server()
        
        logger.info("✅ RL Robot ready!")
        logger.info(f"   Target: {self.target_object}")
        logger.info(f"   Server: {'🟢 Online' if self.server_available else '🔴 Offline (using color detection)'}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        return {
            'server': {
                'url': config.get('server', 'url', fallback='http://192.168.1.100:8000')
            },
            'rl': {
                'learning_rate': config.getfloat('rl', 'alpha', fallback=0.1),
                'discount_factor': config.getfloat('rl', 'gamma', fallback=0.95),
                'epsilon': config.getfloat('rl', 'epsilon', fallback=0.1)
            },
            'target': {
                'object_class': config.get('target', 'object_class', fallback='bottle')
            },
            'safety': {
                'collision_threshold': config.getfloat('safety', 'collision_threshold', fallback=0.15),
                'max_steps': config.getint('safety', 'max_steps_per_episode', fallback=200)
            }
        }
    
    def _load_or_create_qtable(self) -> np.ndarray:
        """Load existing Q-table or create new one"""
        try:
            Q = np.load('q_table.npy')
            logger.info(f"📂 Loaded Q-table from disk (shape: {Q.shape})")
            return Q
        except FileNotFoundError:
            logger.info("📝 Creating new Q-table...")
            return np.zeros((self.num_states, self.num_actions))
    
    def _test_server(self):
        """Test if YOLO server is available"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=1.0)
            self.server_available = response.status_code == 200
        except:
            self.server_available = False
    
    # ========== STATE MANAGEMENT ==========
    
    def get_state(self) -> Tuple[int, Dict]:
        """
        Get current state from sensors and camera
        Returns: (state_id, state_info)
        """
        # Get sensor readings
        distances = self.robot.get_all_distances()
        
        # Get camera frame
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("⚠️ Camera read failed")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detect target
        target = self._detect_target(frame)
        
        # Discretize state
        state_info = {
            'visible': target['visible'],
            'position': self._discretize_position(target),
            'distance': self._discretize_distance(target),
            'ultrasonic': self._discretize_ultrasonic(distances),
            'sensors': distances,
            'target': target
        }
        
        state_id = self._encode_state(state_info)
        return state_id, state_info
    
    def _detect_target(self, frame: np.ndarray) -> Dict:
        """Detect target object using server YOLO or fallback to color"""
        if self.server_available:
            return self._yolo_detection(frame)
        else:
            return self._color_detection(frame)
    
    def _yolo_detection(self, frame: np.ndarray) -> Dict:
        """Request YOLO detection from server"""
        try:
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/detect",
                json={'image': jpeg.tobytes().hex()},
                timeout=0.5
            )
            
            if response.status_code == 200:
                data = response.json()
                detections = data.get('detections', [])
                
                # Find target object
                for det in detections:
                    if det['class'] == self.target_object:
                        bbox = det['bbox']
                        return {
                            'visible': True,
                            'confidence': det['confidence'],
                            'bbox': bbox,
                            'center_x': (bbox[0] + bbox[2]) / 2,
                            'center_y': (bbox[1] + bbox[3]) / 2,
                            'distance': self._estimate_distance_from_bbox(bbox)
                        }
                
                return {'visible': False}
        
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}, falling back to color")
            self.server_available = False
            return self._color_detection(frame)
        
        return {'visible': False}
    
    def _color_detection(self, frame: np.ndarray) -> Dict:
        """Simple color-based detection (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect bright objects (adjust based on your target)
        # Example: detect blue objects
        lower = np.array([100, 100, 100])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:  # Minimum size
                x, y, w, h = cv2.boundingRect(largest)
                return {
                    'visible': True,
                    'confidence': 0.7,
                    'bbox': [x, y, x+w, y+h],
                    'center_x': x + w/2,
                    'center_y': y + h/2,
                    'distance': self._estimate_distance_from_bbox([x, y, x+w, y+h])
                }
        
        return {'visible': False}
    
    def _estimate_distance_from_bbox(self, bbox) -> float:
        """Estimate distance based on bounding box size"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # Simple heuristic: larger bbox = closer object
        if area > 50000:
            return 0.3  # NEAR
        elif area > 10000:
            return 0.7  # MID
        else:
            return 1.5  # FAR
    
    def _discretize_position(self, target: Dict) -> str:
        """Discretize target position: LEFT, CENTER, RIGHT"""
        if not target.get('visible'):
            return 'UNKNOWN'
        
        center_x = target['center_x']
        frame_width = 640
        
        if center_x < frame_width * 0.4:
            return 'LEFT'
        elif center_x < frame_width * 0.6:
            return 'CENTER'
        else:
            return 'RIGHT'
    
    def _discretize_distance(self, target: Dict) -> str:
        """Discretize distance: NEAR, MID, FAR"""
        if not target.get('visible'):
            return 'UNKNOWN'
        
        dist = target.get('distance', 2.0)
        if dist < 0.35:
            return 'NEAR'
        elif dist < 1.0:
            return 'MID'
        else:
            return 'FAR'
    
    def _discretize_ultrasonic(self, distances: Dict) -> str:
        """Discretize ultrasonic: CLEAR, NEAR_OBSTACLE"""
        min_dist = min(distances.values())
        return 'NEAR_OBSTACLE' if min_dist < 0.25 else 'CLEAR'
    
    def _encode_state(self, state_info: Dict) -> int:
        """Encode state info to integer ID"""
        # Encoding: vis(2) × pos(4) × dist(3) × ultra(2)
        vis = 1 if state_info['visible'] else 0
        
        pos_map = {'LEFT': 0, 'CENTER': 1, 'RIGHT': 2, 'UNKNOWN': 3}
        pos = pos_map.get(state_info['position'], 3)
        
        dist_map = {'NEAR': 0, 'MID': 1, 'FAR': 2, 'UNKNOWN': 2}
        dist = dist_map.get(state_info['distance'], 2)
        
        ultra = 1 if state_info['ultrasonic'] == 'NEAR_OBSTACLE' else 0
        
        # Encode to single integer
        state_id = vis * 48 + pos * 12 + dist * 4 + ultra * 2
        return min(state_id, self.num_states - 1)
    
    # ========== Q-LEARNING ==========
    
    def choose_action(self, state_id: int) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            return np.argmax(self.Q[state_id])  # Exploit
    
    def execute_action(self, action: int) -> bool:
        """Execute motor command"""
        if action == 0:  # stop
            self.robot.stop()
            time.sleep(0.1)
            return True
        elif action == 1:  # forward
            return self.robot.move_forward(0.3)
        elif action == 2:  # left
            return self.robot.turn_left(0.3)
        elif action == 3:  # right
            return self.robot.turn_right(0.3)
        return False
    
    def calculate_reward(self, state_info: Dict, next_state_info: Dict, 
                        action: int, action_success: bool) -> float:
        """Calculate reward for transition"""
        reward = -0.1  # Small step penalty
        
        # Goal reached
        if next_state_info['visible'] and next_state_info['distance'] == 'NEAR':
            reward += 100
            logger.info("🎯 TARGET REACHED!")
        
        # Target became visible
        if not state_info['visible'] and next_state_info['visible']:
            reward += 5
        
        # Getting closer
        dist_map = {'NEAR': 0, 'MID': 1, 'FAR': 2, 'UNKNOWN': 2}
        if state_info['visible'] and next_state_info['visible']:
            prev_dist = dist_map[state_info['distance']]
            curr_dist = dist_map[next_state_info['distance']]
            if curr_dist < prev_dist:
                reward += 3  # Moving closer
            elif curr_dist > prev_dist:
                reward -= 1  # Moving away
        
        # Collision penalty
        if not action_success:
            reward -= 50
        
        # Obstacle too close
        if next_state_info['ultrasonic'] == 'NEAR_OBSTACLE' and action == 1:
            reward -= 2
        
        return reward
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """Q-learning update"""
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )
    
    # ========== MAIN LOOP ==========
    
    def run_episode(self) -> float:
        """Run one episode"""
        state_id, state_info = self.get_state()
        episode_reward = 0
        step = 0
        max_steps = self.config['safety']['max_steps']
        
        logger.info(f"📍 Episode {self.episode} started")
        
        while step < max_steps:
            # Choose action
            action = self.choose_action(state_id)
            
            # Execute action
            action_success = self.execute_action(action)
            
            # Observe next state
            next_state_id, next_state_info = self.get_state()
            
            # Calculate reward
            reward = self.calculate_reward(state_info, next_state_info, action, action_success)
            episode_reward += reward
            
            # Update Q-table
            self.update_q_table(state_id, action, reward, next_state_id)
            
            # Log progress
            if step % 20 == 0:
                logger.info(f"Step {step}: Action={self.action_names[action]}, "
                          f"Reward={reward:.1f}, Vis={next_state_info['visible']}")
            
            # Update state
            state_id = next_state_id
            state_info = next_state_info
            step += 1
            self.total_steps += 1
            
            # Check if goal reached
            if reward > 50:  # Goal reached
                break
            
            time.sleep(0.1)  # Control loop rate
        
        logger.info(f"✅ Episode {self.episode} complete: {step} steps, reward={episode_reward:.1f}")
        return episode_reward
    
    def run_forever(self):
        """Main training loop - runs continuously"""
        logger.info("🚀 Starting continuous learning...")
        
        try:
            while True:
                # Run episode
                episode_reward = self.run_episode()
                self.episode_rewards.append(episode_reward)
                
                # Save Q-table periodically
                if self.episode % 10 == 0:
                    np.save('q_table.npy', self.Q)
                    avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                    logger.info(f"💾 Q-table saved | Episode {self.episode} | "
                              f"Avg Reward (100ep): {avg_reward:.1f}")
                
                self.episode += 1
                time.sleep(1)  # Brief pause between episodes
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Training interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down...")
        np.save('q_table.npy', self.Q)
        logger.info("💾 Final Q-table saved")
        self.camera.release()
        self.robot.cleanup()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    robot = MinimalRLRobot()
    robot.run_forever()
