#!/usr/bin/env python3
"""
🤖 RPI MAIN CONTROLLER - Latency-Tolerant VLA+RL Robot

Complete robot controller that handles VLA communication, RL learning,
and motor control for object navigation missions.

Hardware Requirements:
- Raspberry Pi 4
- TB6612FNG motor driver
- 4x HC-SR04 ultrasonic sensors  
- Camera module
- 4 DC motors

Usage:
    sudo python3 rpi_main.py --server-ip <LAPTOP_IP> --target <OBJECT>
    
Example:
    sudo python3 rpi_main.py --server-ip 192.168.1.100 --target bottle
"""

import time
import json
import socket
import struct
import base64
import argparse
import logging
import threading
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pickle
import os
import sys

try:
    import RPi.GPIO as GPIO
    import cv2
    RPI_HARDWARE = True
except ImportError:
    print("⚠️  Running in simulation mode (no RPi hardware)")
    RPI_HARDWARE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================
# HARDWARE CONFIGURATION
# =================================================================

# Motor driver pins (TB6612FNG)
MOTOR_PINS = {
    'AI1': 19, 'AI2': 16, 'PWMA': 13,  # Left motors
    'BI1': 20, 'BI2': 21, 'PWMB': 12,  # Right motors  
    'STBY': 18  # Standby
}

# Ultrasonic sensor pins (HC-SR04) - Updated working configuration
SENSOR_PINS = {
    'FRONT': {'trig': 26, 'echo': 27},   # Previously BACK
    'LEFT':  {'trig': 24, 'echo': 25},
    'RIGHT': {'trig': 5,  'echo': 6}, 
    'BACK':  {'trig': 22, 'echo': 23}    # Previously FRONT
}

# Movement parameters
FORWARD_SPEED = 60
TURN_SPEED = 75
OBSTACLE_THRESHOLD = 0.04  # 4cm
ACTION_DURATION = 0.5  # seconds per action

# =================================================================
# STATE DEFINITIONS
# =================================================================

class RobotAction(Enum):
    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

class NavigationMode(Enum):
    EXPLORATION = "exploration"  # No target found, searching
    NAVIGATION = "navigation"    # Target visible, navigating  
    SEARCH = "search"           # Target lost, searching last known area

# =================================================================
# HARDWARE CONTROLLERS
# =================================================================

class MotorController:
    """TB6612FNG motor driver controller"""
    
    def __init__(self):
        if not RPI_HARDWARE:
            return
            
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup pins
        for pin in MOTOR_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
        
        # Enable driver
        GPIO.output(MOTOR_PINS['STBY'], 1)
        
        # Setup PWM
        self.pwm_left = GPIO.PWM(MOTOR_PINS['PWMA'], 1000)
        self.pwm_right = GPIO.PWM(MOTOR_PINS['PWMB'], 1000) 
        self.pwm_left.start(0)
        self.pwm_right.start(0)
        
        logger.info("Motors initialized")
    
    def _set_left_motor(self, direction: str, speed: int):
        if not RPI_HARDWARE:
            return
        if direction == 'f':
            GPIO.output(MOTOR_PINS['AI1'], 1)
            GPIO.output(MOTOR_PINS['AI2'], 0)
        elif direction == 'b':
            GPIO.output(MOTOR_PINS['AI1'], 0)
            GPIO.output(MOTOR_PINS['AI2'], 1)
        else:
            GPIO.output(MOTOR_PINS['AI1'], 0)
            GPIO.output(MOTOR_PINS['AI2'], 0)
        self.pwm_left.ChangeDutyCycle(speed)
    
    def _set_right_motor(self, direction: str, speed: int):
        if not RPI_HARDWARE:
            return
        if direction == 'f':
            GPIO.output(MOTOR_PINS['BI1'], 1)
            GPIO.output(MOTOR_PINS['BI2'], 0)
        elif direction == 'b':
            GPIO.output(MOTOR_PINS['BI1'], 0)
            GPIO.output(MOTOR_PINS['BI2'], 1)
        else:
            GPIO.output(MOTOR_PINS['BI1'], 0)
            GPIO.output(MOTOR_PINS['BI2'], 0)
        self.pwm_right.ChangeDutyCycle(speed)
    
    def execute_action(self, action: RobotAction, duration: float = ACTION_DURATION):
        """Execute robot action"""
        logger.info(f"Executing: {action.name} for {duration}s")
        
        if action == RobotAction.STOP:
            self._set_left_motor('s', 0)
            self._set_right_motor('s', 0)
        elif action == RobotAction.FORWARD:
            self._set_left_motor('f', FORWARD_SPEED)
            self._set_right_motor('f', FORWARD_SPEED)
        elif action == RobotAction.TURN_LEFT:
            self._set_left_motor('b', TURN_SPEED)
            self._set_right_motor('f', TURN_SPEED)
        elif action == RobotAction.TURN_RIGHT:
            self._set_left_motor('f', TURN_SPEED)
            self._set_right_motor('b', TURN_SPEED)
        
        if action != RobotAction.STOP:
            time.sleep(duration)
            self._set_left_motor('s', 0)
            self._set_right_motor('s', 0)
    
    def backup_for_one_second(self):
        """Back up for exactly 1 second"""
        logger.info("Backing up for 1 second after collision!")
        if not RPI_HARDWARE:
            time.sleep(1.0)
            return
        self._set_left_motor('b', TURN_SPEED)
        self._set_right_motor('b', TURN_SPEED)
        time.sleep(1.0)
        self._set_left_motor('s', 0)
        self._set_right_motor('s', 0)
    
    def cleanup(self):
        if not RPI_HARDWARE:
            return
        self._set_left_motor('s', 0)
        self._set_right_motor('s', 0)
        self.pwm_left.stop()
        self.pwm_right.stop()
        GPIO.cleanup()

class SensorController:
    """HC-SR04 ultrasonic sensor controller (working version)"""
    def __init__(self):
        if not RPI_HARDWARE:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup pins
        for sensor, pins in SENSOR_PINS.items():
            GPIO.setup(pins['trig'], GPIO.OUT)
            GPIO.setup(pins['echo'], GPIO.IN)
            GPIO.output(pins['trig'], 0)
        time.sleep(0.1)  # let sensors settle

    def measure(self, trig, echo):
        """Return distance in cm, or 999 on error."""
        # Trigger pulse
        GPIO.output(trig, 1)
        time.sleep(0.00001)
        GPIO.output(trig, 0)

        # Wait for echo to go HIGH
        timeout = time.time() + 0.02
        while GPIO.input(echo) == 0:
            if time.time() > timeout:
                return 999

        start = time.time()

        # Wait for echo to go LOW
        timeout = time.time() + 0.02
        while GPIO.input(echo) == 1:
            if time.time() > timeout:
                return 999

        end = time.time()

        duration = end - start
        distance = duration * 17150
        return round(distance, 1)

    def read_distance(self, sensor_name: str) -> float:
        if not RPI_HARDWARE:
            return 999
        pins = SENSOR_PINS[sensor_name]
        distance_cm = self.measure(pins['trig'], pins['echo'])
        return distance_cm / 100.0  # convert cm to meters

    def read_all(self) -> Dict[str, float]:
        distances = {}
        for sensor in SENSOR_PINS.keys():
            distances[sensor] = self.read_distance(sensor)
            time.sleep(0.01)
        return distances

    def is_obstacle_ahead(self) -> bool:
        front_distance = self.read_distance('FRONT')
        return front_distance < OBSTACLE_THRESHOLD

# =================================================================
# VLA COMMUNICATION
# =================================================================

class VLAClient:
    """TCP client for VLA server communication"""
    
    def __init__(self, server_ip: str, server_port: int = 9999):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.last_result = None
        self.stats = {
            'requests': 0,
            'successes': 0,
            'avg_response_time': 0
        }
    
    def connect(self) -> bool:
        """Connect to VLA server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            logger.info(f"Connected to VLA server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to VLA server: {e}")
            return False
    
    def capture_image(self) -> Optional[bytes]:
        if not RPI_HARDWARE:
            return b"fake_image_data"

        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

            # Correct Raspberry Pi camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # VALID: enable auto exposure
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # enable auto white balance

            time.sleep(0.5)

            # Clear buffer
            for _ in range(5):
                ret, frame = cap.read()

            cap.release()

            if not ret:
                return None

            # # Optional: very mild correction
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ensure proper color
            # # frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes()

        except Exception as e:
            logger.error(f"Camera error: {e}")
            return None

    
    def query_vla(self, target: str) -> Optional[Dict]:
        """Query VLA server for target detection"""
        if not self.socket:
            if not self.connect():
                return None
        
        # Capture image
        image_bytes = self.capture_image()
        if not image_bytes:
            return None
        
        try:
            start_time = time.time()
            
            # Prepare request
            request = {
                'type': 'vla_query',
                'target': target,
                'image': base64.b64encode(image_bytes).decode('utf-8')
            }
            
            # Send request
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(struct.pack('!I', len(request_data)))
            self.socket.send(request_data)
            
            # Receive response
            length_data = self.socket.recv(4)
            if not length_data:
                return None
            
            response_length = struct.unpack('!I', length_data)[0]
            response_data = b''
            while len(response_data) < response_length:
                chunk = self.socket.recv(min(response_length - len(response_data), 4096))
                if not chunk:
                    break
                response_data += chunk
            
            response = json.loads(response_data.decode('utf-8'))
            response_time = time.time() - start_time
            
            # Update stats
            self.stats['requests'] += 1
            if response.get('success'):
                self.stats['successes'] += 1
            
            self.last_result = response
            
            logger.info(f"VLA query complete: {response_time:.2f}s, found={response.get('target_found', False)}")
            
            return response
            
        except Exception as e:
            logger.error(f"VLA query error: {e}")
            self.socket = None  # Force reconnection next time
            return None

    def change_target(self, new_target: str) -> bool:
        """Change target object on the server"""
        if not self.socket:
            if not self.connect():
                return False
        
        try:
            # Prepare request
            request = {
                'type': 'target_change',
                'target': new_target
            }
            
            # Send request
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(struct.pack('!I', len(request_data)))
            self.socket.send(request_data)
            
            # Receive response
            length_data = self.socket.recv(4)
            if not length_data:
                return False
            
            response_length = struct.unpack('!I', length_data)[0]
            response_data = b''
            while len(response_data) < response_length:
                chunk = self.socket.recv(min(response_length - len(response_data), 4096))
                if not chunk:
                    break
                response_data += chunk
            
            response = json.loads(response_data.decode('utf-8'))
            
            if response.get('success'):
                logger.info(f"Target changed to: {response.get('new_target')}")
                return True
            else:
                logger.error("Failed to change target on server")
                return False
                
        except Exception as e:
            logger.error(f"Target change error: {e}")
            self.socket = None  # Force reconnection next time
            return False

# =================================================================
# RL AGENT
# =================================================================

class QLearningAgent:
    """Q-Learning agent for robot navigation"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_count = 0
        self.total_reward = 0
        self.action_counts = defaultdict(int)
        
        # Load saved Q-table if exists
        self.load_q_table()
    
    def get_state(self, vla_result: Optional[Dict], sensors: Dict[str, float], mode: NavigationMode) -> str:
        """Create state representation"""
        # VLA information
        target_visible = vla_result is not None and vla_result.get('target_found', False)
        target_position = vla_result.get('position', 'unknown') if target_visible else 'unknown'
        target_distance = vla_result.get('distance', 'unknown') if target_visible else 'unknown'
        
        # Sensor information
        obstacle_front = sensors['FRONT'] < OBSTACLE_THRESHOLD
        obstacle_left = sensors['LEFT'] < OBSTACLE_THRESHOLD
        obstacle_right = sensors['RIGHT'] < OBSTACLE_THRESHOLD
        
        # Create state string
        state = f"{mode.value}_{target_visible}_{target_position}_{target_distance}_{obstacle_front}_{obstacle_left}_{obstacle_right}"
        return state
    
    def choose_action(self, state: str, valid_actions: List[RobotAction] = None) -> RobotAction:
        """Choose action using epsilon-greedy policy"""
        if valid_actions is None:
            valid_actions = list(RobotAction)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
            logger.debug(f"🎲 Random action: {action.name}")
        else:
            # Choose best action
            q_values = {action: self.q_table[state][action.value] for action in valid_actions}
            action = max(q_values, key=q_values.get)
            logger.debug(f"🧠 Best action: {action.name} (Q={q_values[action]:.2f})")
        
        self.action_counts[action] += 1
        return action
    
    def update_q_value(self, state: str, action: RobotAction, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action.value]
        
        # Find maximum Q-value for next state
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action.value] = new_q
        
        logger.debug(f"📚 Q-update: {state[:20]}... -> {action.name}: {current_q:.3f} -> {new_q:.3f}")
    
    def calculate_reward(self, vla_result: Optional[Dict], sensors: Dict[str, float], 
                        action: RobotAction, previous_vla: Optional[Dict]) -> float:
        """Calculate reward for current situation"""
        reward = 0
        
        # Base step penalty
        reward -= 0.1
        
        # Collision penalty
        if sensors['FRONT'] < OBSTACLE_THRESHOLD and action == RobotAction.FORWARD:
            reward -= 10
            logger.debug("💥 Collision penalty")
        
        # Target visibility rewards
        if vla_result and vla_result.get('target_found'):
            reward += 2  # Found target bonus
            
            # Position improvement rewards
            position = vla_result.get('position')
            if position == 'center':
                reward += 5  # Centered on target
            elif position in ['left', 'right']:
                reward += 1  # Target visible but not centered
            
            # Distance improvement (if we have previous measurement)
            if previous_vla and previous_vla.get('target_found'):
                prev_dist = previous_vla.get('distance')
                curr_dist = vla_result.get('distance')
                if prev_dist == 'far' and curr_dist == 'near':
                    reward += 3  # Got closer
                elif prev_dist == 'near' and curr_dist == 'far':
                    reward -= 1  # Got farther
            
            # Very close to target
            if vla_result.get('distance') == 'near' and vla_result.get('position') == 'center':
                reward += 20  # Mission success!
                logger.info("🎯 TARGET REACHED!")
        
        logger.debug(f"🎁 Reward: {reward:.2f}")
        return reward
    
    def end_episode(self):
        """End current episode and update parameters"""
        self.episode_count += 1
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Save Q-table periodically
        if self.episode_count % 10 == 0:
            self.save_q_table()
        
        logger.info(f"Episode {self.episode_count} complete, epsilon={self.epsilon:.3f}")
    
    def save_q_table(self, filename: str = "q_table.pkl"):
        """Save Q-table to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'q_table': dict(self.q_table),
                    'episode_count': self.episode_count,
                    'epsilon': self.epsilon,
                    'stats': self.action_counts
                }, f)
            logger.info(f"Q-table saved ({len(self.q_table)} states)")
        except Exception as e:
            logger.error(f"Failed to save Q-table: {e}")
    
    def load_q_table(self, filename: str = "q_table.pkl"):
        """Load Q-table from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
                self.episode_count = data.get('episode_count', 0)
                self.epsilon = data.get('epsilon', self.epsilon)
                self.action_counts = defaultdict(int, data.get('stats', {}))
                logger.info(f"Q-table loaded ({len(self.q_table)} states, episode {self.episode_count})")
        except Exception as e:
            logger.info(f"No saved Q-table found, starting fresh")

# =================================================================
# MAIN ROBOT CONTROLLER
# =================================================================

class RobotController:
    """Main robot controller integrating VLA and RL"""
    
    def __init__(self, server_ip: str, target: str):
        self.server_ip = server_ip
        self.target = target
        
        # Initialize hardware
        self.motors = MotorController()
        self.sensors = SensorController()
        self.vla_client = VLAClient(server_ip)
        self.rl_agent = QLearningAgent()
        
        # State tracking
        self.mode = NavigationMode.EXPLORATION
        self.last_vla_result = None
        self.last_vla_time = 0
        self.steps_since_vla = 0
        self.mission_start_time = time.time()
        
        # Stats
        self.total_steps = 0
        self.successful_missions = 0
        
    def change_target(self, new_target: str) -> bool:
        """Change target object for both robot and server"""
        logger.info(f"Changing target from '{self.target}' to '{new_target}'")
        
        # Change target on server first
        if self.vla_client.change_target(new_target):
            # If server change successful, update local target
            self.target = new_target
            logger.info(f"Target successfully changed to '{self.target}'")
            return True
        else:
            logger.error(f"Failed to change target to '{new_target}'")
            return False
        
    def determine_navigation_mode(self, vla_result: Optional[Dict]) -> NavigationMode:
        """Determine current navigation mode"""
        if vla_result and vla_result.get('target_found'):
            return NavigationMode.NAVIGATION
        elif self.last_vla_result and self.last_vla_result.get('target_found'):
            return NavigationMode.SEARCH  # Lost target recently
        else:
            return NavigationMode.EXPLORATION
    
    def get_valid_actions(self, sensors: Dict[str, float]) -> List[RobotAction]:
        """Get valid actions based on sensor readings"""
        valid_actions = [RobotAction.STOP, RobotAction.TURN_LEFT, RobotAction.TURN_RIGHT]
        
        # Only allow forward if no obstacle ahead
        if sensors['FRONT'] > OBSTACLE_THRESHOLD:
            valid_actions.append(RobotAction.FORWARD)
        
        return valid_actions
    
    def should_query_vla(self) -> bool:
        """Decide if we should query VLA"""
        time_since_last = time.time() - self.last_vla_time
        
        # Query VLA if:
        # 1. Never queried before
        # 2. Been too long since last query (6 seconds max)
        # 3. In exploration mode every 3 seconds (maximize 360° coverage)
        # 4. In search mode every 2 seconds (target nearby)
        # 5. In navigation mode every 4 seconds (verify target still visible)
        
        if self.last_vla_time == 0:
            return True
        elif time_since_last > 6:  # Never go too long without checking
            return True
        elif self.mode == NavigationMode.EXPLORATION and time_since_last > 3:
            return True
        elif self.mode == NavigationMode.SEARCH and time_since_last > 2:
            return True
        elif self.mode == NavigationMode.NAVIGATION and time_since_last > 4:
            return True
        else:
            return False
    
    def run_mission(self, max_steps: int = 1000, max_time: int = 300):
        """Run complete mission"""
        logger.info(f"Starting mission: Find {self.target}")
        logger.info(f"Episode {self.rl_agent.episode_count + 1}")
        
        step = 0
        previous_vla = None
        
        try:
            while step < max_steps:
                step += 1
                self.total_steps += 1
                step_start_time = time.time()
                
                # Check mission timeout
                if time.time() - self.mission_start_time > max_time:
                    logger.info("Mission timeout")
                    break
                
                # Read sensors
                sensors = self.sensors.read_all()
                logger.info(f"Sensors: F={sensors['FRONT']:.2f}m, L={sensors['LEFT']:.2f}m, R={sensors['RIGHT']:.2f}m")
                
                # Query VLA if needed
                current_vla = self.last_vla_result
                if self.should_query_vla():
                    logger.info("Querying VLA...")
                    vla_result = self.vla_client.query_vla(self.target)
                    if vla_result and vla_result.get('success'):
                        current_vla = vla_result
                        self.last_vla_result = vla_result
                        self.last_vla_time = time.time()
                        self.steps_since_vla = 0
                    else:
                        logger.warning("VLA query failed")
                
                # Update navigation mode
                self.mode = self.determine_navigation_mode(current_vla)
                logger.info(f"Mode: {self.mode.value}")
                
                # Create state for RL
                state = self.rl_agent.get_state(current_vla, sensors, self.mode)
                
                # Get valid actions
                valid_actions = self.get_valid_actions(sensors)
                
                # Choose action
                action = self.rl_agent.choose_action(state, valid_actions)

                # PREVENTIVE: Check for immediate collision before moving forward
                if action == RobotAction.FORWARD and sensors['FRONT'] < OBSTACLE_THRESHOLD:
                    logger.warning(f"PREVENTED COLLISION: Front={sensors['FRONT']:.3f}m < {OBSTACLE_THRESHOLD}m")
                    action = RobotAction.STOP  # Override action to stop
                    new_sensors = sensors  # Use current sensors, no movement
                else:
                    # Execute action
                    logger.debug(f"🚗 Executing {action.name}, Front={sensors['FRONT']:.3f}m")
                    self.motors.execute_action(action)
                    new_sensors = self.sensors.read_all()
                    # Check for collision after action and back off if needed
                    if new_sensors['FRONT'] < OBSTACLE_THRESHOLD:
                        logger.warning(f"COLLISION DETECTED: Front={new_sensors['FRONT']:.3f}m < {OBSTACLE_THRESHOLD}m")
                        self.motors.backup_for_one_second()  # <-- backup for 1 second
                        new_sensors = self.sensors.read_all()  # Re-read after backing off
                
                # Calculate reward
                reward = self.rl_agent.calculate_reward(current_vla, new_sensors, action, previous_vla)
                
                # Update Q-learning
                new_state = self.rl_agent.get_state(current_vla, new_sensors, self.mode)
                self.rl_agent.update_q_value(state, action, reward, new_state)
                
                # Check for mission success
                mission_success = False
                
                # Method 1: VLA confirms target is centered and near
                if (current_vla and current_vla.get('target_found') and 
                    current_vla.get('distance') == 'near' and 
                    current_vla.get('position') == 'center'):
                    mission_success = True
                    logger.info("TARGET REACHED (VLA confirmed)")
                
                # Method 2: Physical collision while target was recently visible
                elif (new_sensors['FRONT'] < OBSTACLE_THRESHOLD and 
                      action == RobotAction.FORWARD and
                      current_vla and current_vla.get('target_found')):
                    mission_success = True
                    logger.info("TARGET REACHED (Physical collision)")
                
                # Method 3: Very close obstacle while target visible and centered
                elif (new_sensors['FRONT'] < 0.10 and  # 10cm threshold
                      current_vla and current_vla.get('target_found') and
                      current_vla.get('position') == 'center'):
                    mission_success = True
                    logger.info("TARGET REACHED (Close approach)")
                
                if mission_success:
                    logger.info("MISSION SUCCESSFUL!")
                    self.successful_missions += 1
                    reward += 50  # Extra success bonus
                    break
                
                # Update tracking
                previous_vla = current_vla
                self.steps_since_vla += 1
                
                # Step timing info
                step_time = time.time() - step_start_time
                logger.info(f"Step {step} completed in {step_time:.2f}s\n")
                
                # Small delay between steps
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Mission interrupted by user")
        
        # End episode
        self.rl_agent.end_episode()
        
        # Final stats
        mission_time = time.time() - self.mission_start_time
        logger.info(f"Mission complete: {step} steps in {mission_time:.1f}s")
        logger.info(f"Success rate: {self.successful_missions}/{self.rl_agent.episode_count} ({self.successful_missions/max(1,self.rl_agent.episode_count)*100:.1f}%)")
    
    def cleanup(self):
        """Cleanup hardware resources"""
        logger.info("Cleaning up...")
        self.motors.cleanup()
        self.rl_agent.save_q_table()

# =================================================================
# MAIN FUNCTION
# =================================================================

def main():
    """Main function"""
    # Hardcoded server IP - change this to your laptop's IP
    DEFAULT_SERVER_IP = "10.28.185.229"  # Change this to your actual laptop IP
    
    parser = argparse.ArgumentParser(description='VLA+RL Robot Controller')
    parser.add_argument('--server-ip', default=DEFAULT_SERVER_IP, help='VLA server IP address')
    parser.add_argument('--target', default='bottle', help='Target object to find')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    args = parser.parse_args()
    
    print("VLA+RL ROBOT CONTROLLER")
    print("=" * 50)
    print(f"Target: {args.target}")
    print(f"VLA Server: {args.server_ip}")
    print(f"Episodes: {args.episodes}")
    print("=" * 50)
    
    robot = None
    try:
        robot = RobotController(args.server_ip, args.target)
        
        for episode in range(args.episodes):
            logger.info(f"\nEPISODE {episode + 1}/{args.episodes}")
            robot.run_mission()
            
            # Short break between episodes
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if robot:
            robot.cleanup()

if __name__ == "__main__":
    main()