#!/usr/bin/env python3
"""
Hierarchical RL Main Controller
Combines VLA + Q-Learning with 3-layer hierarchical control:
- LAYER 1: Survival/Safety (Fast, 10-50Hz, ultrasonics only)
- LAYER 2: Exploration (Mid, 1-2Hz, when target not visible)
- LAYER 3: Goal Seeking (Slow, 1fps, Q-learning with VLA when target visible)

State Machine:
- TARGET_VISIBLE: Use Q-learning to approach target
- SEARCH_ROTATE: Rotate in place to reacquire lost target
- EXPLORATION: Wall-following, doorway detection, frontier exploration
- OBSTACLE_AVOID: Triggered by safety layer, overrides everything

Integrates with remote server's pin diagram and web interface
"""

import sys
import os
import time
import threading
import logging
import json
import numpy as np
from enum import Enum
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import hardware controller (from remote server with correct pin diagram)
from hardware import RobotHardware

# Import VLA components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../EDAI/vla_mvp'))
from vla import VLAController, DetectedObject

# Import Q-Learning components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../EDAI'))
from model1 import (
    QLearningAgent, State, Visibility, Position, DistanceBin, 
    UltrasonicBin, LastSeenDirection, Action as QLAction, VLAProcessor, CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =================================================================
# HIERARCHICAL STATE MACHINE
# =================================================================

class RobotState(Enum):
    """High-level state machine states"""
    TARGET_VISIBLE = "TARGET_VISIBLE"       # Using Q-learning to approach
    SEARCH_ROTATE = "SEARCH_ROTATE"         # Rotating to reacquire target
    EXPLORATION = "EXPLORATION"             # Exploring new areas
    OBSTACLE_AVOID = "OBSTACLE_AVOID"       # Emergency obstacle avoidance
    STOPPED = "STOPPED"                     # Manual stop

class ExplorationMode(Enum):
    """Sub-modes for exploration state"""
    WALL_FOLLOW_LEFT = "WALL_FOLLOW_LEFT"
    WALL_FOLLOW_RIGHT = "WALL_FOLLOW_RIGHT"
    DOORWAY_DETECT = "DOORWAY_DETECT"
    FRONTIER_EXPLORE = "FRONTIER_EXPLORE"
    RANDOM_WALK = "RANDOM_WALK"
    ESCAPE_STUCK = "ESCAPE_STUCK"

# =================================================================
# LAYER 1: SURVIVAL/SAFETY (FAST - 10-50Hz)
# =================================================================

class SafetyLayer:
    """
    Layer 1: Survival/Safety
    - Runs at 10-50 Hz
    - Uses ONLY ultrasonics
    - Can override ANY other command
    - Priority: Don't crash
    """
    
    def __init__(self, hardware: RobotHardware):
        self.hardware = hardware
        self.enabled = True
        
        # Safety thresholds (meters)
        self.COLLISION_THRESH = 0.15      # Hard stop
        self.SAFETY_THRESH = 0.20         # Caution zone
        self.SIDE_THRESH = 0.12           # Side obstacle threshold
        self.BACK_THRESH = 0.10           # Backing up threshold
        
        # State tracking
        self.unsafe = False
        self.last_check_time = time.time()
        self.check_interval = 0.02  # 50Hz
        
        logger.info("Safety Layer initialized (Layer 1)")
    
    def check_safety(self) -> Tuple[bool, str]:
        """
        Check if current state is safe
        Returns: (is_safe, reason)
        """
        if not self.enabled:
            return True, "Safety disabled"
        
        # Get ultrasonic readings
        distances = self.hardware.get_all_distances()
        
        # CRITICAL: Front collision imminent
        if distances['front'] is not None and distances['front'] < self.COLLISION_THRESH:
            self.unsafe = True
            return False, f"COLLISION IMMINENT: Front {distances['front']:.2f}m"
        
        # Left/Right too close
        if distances['left'] is not None and distances['left'] < self.SIDE_THRESH:
            self.unsafe = True
            return False, f"LEFT OBSTACLE: {distances['left']:.2f}m"
        
        if distances['right'] is not None and distances['right'] < self.SIDE_THRESH:
            self.unsafe = True
            return False, f"RIGHT OBSTACLE: {distances['right']:.2f}m"
        
        # Check if trapped (all sides blocked)
        blocked_sides = 0
        if distances['front'] and distances['front'] < self.SAFETY_THRESH:
            blocked_sides += 1
        if distances['left'] and distances['left'] < self.SAFETY_THRESH:
            blocked_sides += 1
        if distances['right'] and distances['right'] < self.SAFETY_THRESH:
            blocked_sides += 1
        if distances['back'] and distances['back'] < self.BACK_THRESH:
            blocked_sides += 1
        
        if blocked_sides >= 3:
            self.unsafe = True
            return False, "TRAPPED: 3+ sides blocked"
        
        self.unsafe = False
        return True, "OK"
    
    def override_action(self, intended_action: str, distances: dict) -> str:
        """
        Override intended action if unsafe
        Returns: safe_action (may be different from intended)
        """
        is_safe, reason = self.check_safety()
        
        if is_safe:
            return intended_action
        
        logger.warning(f"SAFETY OVERRIDE: {reason}")
        
        # Emergency response logic
        if distances['front'] and distances['front'] < self.COLLISION_THRESH:
            if distances['back'] and distances['back'] > 0.3:
                return "backward"
            elif distances['left'] and distances['right']:
                # Turn away from closer side
                if distances['left'] > distances['right']:
                    return "turn_left"
                else:
                    return "turn_right"
            else:
                return "stop"
        
        # Side obstacles
        if distances['left'] and distances['left'] < self.SIDE_THRESH:
            return "turn_right"
        if distances['right'] and distances['right'] < self.SIDE_THRESH:
            return "turn_left"
        
        # Default: stop
        return "stop"

# =================================================================
# LAYER 2: EXPLORATION/NAVIGATION (MID - 1-2Hz)
# =================================================================

class ExplorationLayer:
    """
    Layer 2: Exploration/Navigation
    - Runs at 1-2 Hz
    - Used when camera does NOT see target
    - Classic robotics logic (not RL)
    - Behaviors: wall-following, doorway detection, frontier exploration
    """
    
    def __init__(self, hardware: RobotHardware):
        self.hardware = hardware
        self.mode = ExplorationMode.WALL_FOLLOW_LEFT
        
        # State tracking
        self.stuck_counter = 0
        self.stuck_threshold = 10  # steps
        self.last_position = None
        self.position_history = deque(maxlen=5)
        
        # Wall following parameters
        self.wall_follow_distance = 0.30  # meters - target distance from wall
        self.wall_follow_tolerance = 0.10  # meters - acceptable variation
        
        # Doorway detection
        self.last_front_distance = None
        self.doorway_drop_threshold = 0.5  # meter increase suggests doorway
        
        # Exploration counters
        self.exploration_steps = 0
        self.mode_switch_interval = 50  # steps before switching exploration mode
        
        logger.info("Exploration Layer initialized (Layer 2)")
    
    def decide_exploration_action(self, distances: dict) -> str:
        """
        Decide exploration action based on current mode and sensor readings
        """
        self.exploration_steps += 1
        
        # Check if stuck
        if self._is_stuck():
            self.mode = ExplorationMode.ESCAPE_STUCK
            self.stuck_counter += 1
        
        # Execute mode-specific behavior
        if self.mode == ExplorationMode.WALL_FOLLOW_LEFT:
            return self._wall_follow_left(distances)
        
        elif self.mode == ExplorationMode.WALL_FOLLOW_RIGHT:
            return self._wall_follow_right(distances)
        
        elif self.mode == ExplorationMode.DOORWAY_DETECT:
            return self._doorway_detect(distances)
        
        elif self.mode == ExplorationMode.ESCAPE_STUCK:
            return self._escape_stuck(distances)
        
        elif self.mode == ExplorationMode.RANDOM_WALK:
            return self._random_walk(distances)
        
        else:
            return self._wall_follow_left(distances)  # Default
    
    def _wall_follow_left(self, distances: dict) -> str:
        """Follow left wall"""
        left_dist = distances.get('left')
        front_dist = distances.get('front')
        
        if front_dist and front_dist < 0.4:
            # Obstacle ahead, turn right
            return "turn_right"
        
        if left_dist is None:
            # No left wall, turn left to find it
            return "turn_left"
        
        if left_dist < self.wall_follow_distance - self.wall_follow_tolerance:
            # Too close to wall, turn right
            return "turn_right"
        
        elif left_dist > self.wall_follow_distance + self.wall_follow_tolerance:
            # Too far from wall, turn left
            return "turn_left"
        
        else:
            # Good distance, go forward
            return "forward"
    
    def _wall_follow_right(self, distances: dict) -> str:
        """Follow right wall (mirror of left)"""
        right_dist = distances.get('right')
        front_dist = distances.get('front')
        
        if front_dist and front_dist < 0.4:
            return "turn_left"
        
        if right_dist is None:
            return "turn_right"
        
        if right_dist < self.wall_follow_distance - self.wall_follow_tolerance:
            return "turn_left"
        
        elif right_dist > self.wall_follow_distance + self.wall_follow_tolerance:
            return "turn_right"
        
        else:
            return "forward"
    
    def _doorway_detect(self, distances: dict) -> str:
        """Detect and enter doorways"""
        front_dist = distances.get('front')
        
        # Check for sudden distance increase (doorway)
        if self.last_front_distance and front_dist:
            distance_change = front_dist - self.last_front_distance
            
            if distance_change > self.doorway_drop_threshold:
                logger.info(f"DOORWAY DETECTED: {distance_change:.2f}m increase")
                # Go through doorway
                self.last_front_distance = front_dist
                return "forward"
        
        self.last_front_distance = front_dist
        
        # Default: continue wall following
        return self._wall_follow_left(distances)
    
    def _escape_stuck(self, distances: dict) -> str:
        """Escape from stuck situation"""
        # Try rotating 360 degrees to scan
        if self.stuck_counter % 4 == 0:
            return "turn_right"
        elif self.stuck_counter % 4 == 1:
            return "turn_right"
        elif self.stuck_counter % 4 == 2:
            # Try backing up
            if distances.get('back', 1.0) > 0.3:
                return "backward"
            else:
                return "turn_left"
        else:
            self.stuck_counter = 0
            self.mode = ExplorationMode.WALL_FOLLOW_LEFT
            return "forward"
    
    def _random_walk(self, distances: dict) -> str:
        """Random walk with bias towards open space"""
        import random
        
        # Find most open direction
        front = distances.get('front', 0)
        left = distances.get('left', 0)
        right = distances.get('right', 0)
        
        max_dist = max(front or 0, left or 0, right or 0)
        
        if max_dist == front and front > 0.5:
            return "forward"
        elif max_dist == left:
            return "turn_left"
        elif max_dist == right:
            return "turn_right"
        else:
            return random.choice(["turn_left", "turn_right"])
    
    def _is_stuck(self) -> bool:
        """Detect if robot is stuck"""
        # Simple stuck detection: if very little movement in last N steps
        # (In real implementation, would use odometry/encoders)
        return self.stuck_counter > self.stuck_threshold

# =================================================================
# LAYER 3: GOAL SEEKING (SLOW - 1fps, Q-LEARNING)
# =================================================================

class GoalSeekingLayer:
    """
    Layer 3: Goal Seeking
    - Runs at ~1 fps
    - Uses Q-learning for target approach
    - ONLY active when target is visible
    - Preserves original VLA reward system
    """
    
    def __init__(self, vla_controller: VLAController, hardware: RobotHardware):
        self.vla = vla_controller
        self.hardware = hardware
        self.agent = QLearningAgent()
        self.vla_processor = VLAProcessor()
        
        # Load existing Q-table if available
        self._load_q_table()
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.episode_reward = 0
        self.episode_steps = 0
        
        # Target tracking for reward calculation
        self.target_class = "bottle"  # From config.ini
        self.last_target_visible = False
        self.last_target_centered = False
        self.last_distance_bin = None
        
        # Frame center for centering reward
        self.frame_center_x = 320  # 640/2
        self.frame_center_y = 240  # 480/2
        self.center_margin = 50  # pixels - margin of error for "centered"
        
        logger.info("Goal Seeking Layer initialized (Layer 3 - Q-Learning)")
    
    def decide_goal_seeking_action(self, target_detected: DetectedObject, 
                                   distances: dict) -> Tuple[str, float]:
        """
        Decide action using Q-learning when target is visible
        Returns: (action, reward)
        """
        # Get current state from VLA and ultrasonic data
        current_state = self._get_state(target_detected, distances)
        
        # Calculate reward (preserves original VLA reward system)
        reward = self._calculate_reward(target_detected, distances)
        
        # Update Q-table if we have a previous action
        if self.current_state and self.last_action:
            self.agent.update_q_value(
                self.current_state, 
                self.last_action, 
                reward,
                current_state,
                done=False  # Episode continues
            )
        
        # Choose next action using epsilon-greedy
        q_action = self.agent.choose_action(current_state)
        
        # Convert Q-learning action to motor command
        motor_action = self._q_action_to_motor_command(q_action)
        
        # Update tracking
        self.current_state = current_state
        self.last_action = q_action
        self.episode_reward += reward
        self.episode_steps += 1
        
        return motor_action, reward
    
    def _get_state(self, target: DetectedObject, distances: dict) -> State:
        """Build state from VLA detection and ultrasonic readings"""
        # VLA processing
        vla_data = {
            'confidence': target.confidence if target else 0,
            'bbox': target.bbox if target else [0, 0, 0, 0],
            'distance': target.distance if target else None
        }
        
        vis, pos, dist_bin = self.vla_processor.process_vla_output(vla_data)
        
        # Ultrasonic processing
        ultra_bin = self._get_ultrasonic_bin(distances)
        
        # Last seen direction
        if vis == Visibility.VISIBLE:
            if pos == Position.LEFT:
                last_seen = LastSeenDirection.LAST_L
            elif pos == Position.RIGHT:
                last_seen = LastSeenDirection.LAST_R
            else:
                last_seen = LastSeenDirection.LAST_C
        else:
            last_seen = LastSeenDirection.NONE
        
        return State(vis, pos, dist_bin, ultra_bin, last_seen)
    
    def _get_ultrasonic_bin(self, distances: dict) -> UltrasonicBin:
        """Determine ultrasonic bin from sensor readings"""
        front = distances.get('front', 10)
        left = distances.get('left', 10)
        right = distances.get('right', 10)
        
        min_dist = min(front or 10, left or 10, right or 10)
        
        if min_dist < CONFIG.ULTRASONIC_NEAR_THRESH:
            return UltrasonicBin.NEAR_OBSTACLE
        else:
            return UltrasonicBin.CLEAR
    
    def _calculate_reward(self, target: DetectedObject, distances: dict) -> float:
        """
        Calculate reward preserving original VLA reward system:
        - +reward for target in frame
        - +reward for centering target vertically (with margin of error)
        - -punishment for no target
        - -punishment for collision risk
        """
        reward = 0
        
        # Base step penalty
        reward += CONFIG.STEP_PENALTY
        
        # TARGET IN FRAME REWARD
        target_visible = target is not None and target.confidence > CONFIG.VLA_CONFIDENCE_THRESH
        
        if target_visible:
            # Reward for having target in frame
            if not self.last_target_visible:
                # Bonus for regaining visibility (0→1 transition)
                reward += CONFIG.VIS_TRANSITION_REWARD
                logger.info(f"TARGET ACQUIRED: +{CONFIG.VIS_TRANSITION_REWARD} reward")
            else:
                # Small reward for maintaining visibility
                reward += 0.5
            
            # CENTERING REWARD (vertical centering as per original)
            # Check if target center is within margin of frame center
            target_center_x = target.center_x
            target_center_y = target.center_y
            
            x_error = abs(target_center_x - self.frame_center_x)
            y_error = abs(target_center_y - self.frame_center_y)
            
            # Reward for vertical centering (Y-axis)
            if y_error < self.center_margin:
                if not self.last_target_centered:
                    reward += CONFIG.BEARING_IMPROVEMENT_REWARD
                    logger.info(f"TARGET CENTERED: +{CONFIG.BEARING_IMPROVEMENT_REWARD} reward")
                else:
                    reward += 1  # Maintain centering
                self.last_target_centered = True
            else:
                # Punish for moving away from center
                if self.last_target_centered:
                    reward += CONFIG.BEARING_WORSENING_PENALTY
                self.last_target_centered = False
            
            # DISTANCE-BASED REWARD
            if target.distance:
                current_dist_bin = self._distance_to_bin(target.distance)
                
                # Reward for getting closer
                if self.last_distance_bin and current_dist_bin != self.last_distance_bin:
                    if (self.last_distance_bin == DistanceBin.FAR and 
                        current_dist_bin in [DistanceBin.MID, DistanceBin.NEAR]):
                        reward += CONFIG.DISTANCE_IMPROVEMENT_REWARD
                    elif (self.last_distance_bin == DistanceBin.MID and 
                          current_dist_bin == DistanceBin.NEAR):
                        reward += CONFIG.DISTANCE_IMPROVEMENT_REWARD
                    elif (self.last_distance_bin == DistanceBin.NEAR and 
                          current_dist_bin in [DistanceBin.MID, DistanceBin.FAR]):
                        # Moving away is bad
                        reward += CONFIG.DISTANCE_WORSENING_PENALTY
                
                self.last_distance_bin = current_dist_bin
                
                # GOAL REACHED
                if target.distance < 0.35:  # NEAR threshold
                    reward += CONFIG.GOAL_REACHED_REWARD
                    logger.info(f"GOAL REACHED: +{CONFIG.GOAL_REACHED_REWARD} reward!")
        
        else:
            # NO TARGET IN FRAME PUNISHMENT
            if self.last_target_visible:
                logger.warning("TARGET LOST: -2 reward")
                reward -= 2
            self.last_target_visible = False
            self.last_target_centered = False
        
        # COLLISION RISK PUNISHMENT
        front_dist = distances.get('front')
        if front_dist and front_dist < CONFIG.COLLISION_THRESH:
            reward += CONFIG.COLLISION_PENALTY
            logger.warning(f"COLLISION IMMINENT: {CONFIG.COLLISION_PENALTY} reward")
        elif front_dist and front_dist < CONFIG.SAFETY_THRESH:
            reward += CONFIG.FORWARD_NEAR_OBSTACLE_PENALTY
        
        # Update tracking
        self.last_target_visible = target_visible
        
        return reward
    
    def _distance_to_bin(self, distance: float) -> DistanceBin:
        """Convert distance to bin"""
        if distance <= CONFIG.DISTANCE_NEAR_THRESH:
            return DistanceBin.NEAR
        elif distance <= CONFIG.DISTANCE_MID_THRESH:
            return DistanceBin.MID
        else:
            return DistanceBin.FAR
    
    def _q_action_to_motor_command(self, q_action: QLAction) -> str:
        """Convert Q-learning action to motor command"""
        if q_action == QLAction.STOP:
            return "stop"
        elif q_action == QLAction.FORWARD:
            return "forward"
        elif q_action == QLAction.TURN_LEFT:
            return "turn_left"
        elif q_action == QLAction.TURN_RIGHT:
            return "turn_right"
        else:
            return "stop"
    
    def _load_q_table(self):
        """Load existing Q-table if available"""
        q_table_path = "model1_checkpoint.json"
        if os.path.exists(q_table_path):
            self.agent.load_q_table(q_table_path)
            logger.info(f"Loaded Q-table with {len(self.agent.q_table)} states")
        else:
            logger.info("Starting with empty Q-table")
    
    def save_q_table(self):
        """Save Q-table to disk"""
        q_table_path = "model1_checkpoint.json"
        self.agent.save_q_table(q_table_path)
        logger.info(f"Saved Q-table with {len(self.agent.q_table)} states")

# =================================================================
# MAIN HIERARCHICAL CONTROLLER
# =================================================================

class HierarchicalRLController:
    """
    Main hierarchical controller coordinating all 3 layers
    Implements state machine for high-level behaviors
    """
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("HIERARCHICAL RL CONTROLLER - INITIALIZING")
        logger.info("=" * 60)
        
        # Initialize hardware
        self.hardware = RobotHardware()
        
        # Initialize VLA
        self.vla = VLAController(camera_id=0)
        self.vla.start()
        time.sleep(2)  # Wait for camera to initialize
        
        # Initialize layers
        self.safety_layer = SafetyLayer(self.hardware)
        self.exploration_layer = ExplorationLayer(self.hardware)
        self.goal_seeking_layer = GoalSeekingLayer(self.vla, self.hardware)
        
        # State machine
        self.current_state = RobotState.SEARCH_ROTATE
        self.previous_state = None
        
        # Target tracking
        self.target_class = "bottle"  # From config
        self.target_lost_counter = 0
        self.target_lost_threshold = 5  # frames before switching to SEARCH
        self.search_rotation_counter = 0
        self.search_rotation_limit = 12  # 12 x 30° = 360°
        
        # Control loop
        self.running = False
        self.control_thread = None
        
        # Performance tracking
        self.loop_times = deque(maxlen=100)
        self.state_transition_log = []
        
        logger.info("Hierarchical RL Controller initialized")
        logger.info("=" * 60)
    
    def start(self):
        """Start the control loop"""
        if self.running:
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("Control loop started")
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2)
        self.hardware.stop()
        self.vla.stop()
        self.goal_seeking_layer.save_q_table()
        logger.info("Control loop stopped")
    
    def _control_loop(self):
        """Main control loop coordinating all layers"""
        logger.info("Entering main control loop")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get sensor readings (LAYER 1 - fast)
                distances = self.hardware.get_all_distances()
                
                # LAYER 1: Safety check (ALWAYS runs first, can override anything)
                is_safe, safety_reason = self.safety_layer.check_safety()
                
                if not is_safe:
                    # Force OBSTACLE_AVOID state
                    self._transition_to_state(RobotState.OBSTACLE_AVOID, safety_reason)
                
                # Execute state machine
                if self.current_state == RobotState.OBSTACLE_AVOID:
                    action = self._handle_obstacle_avoid(distances)
                
                elif self.current_state == RobotState.TARGET_VISIBLE:
                    action = self._handle_target_visible(distances)
                
                elif self.current_state == RobotState.SEARCH_ROTATE:
                    action = self._handle_search_rotate(distances)
                
                elif self.current_state == RobotState.EXPLORATION:
                    action = self._handle_exploration(distances)
                
                elif self.current_state == RobotState.STOPPED:
                    action = "stop"
                
                else:
                    action = "stop"
                
                # LAYER 1: Final safety override
                final_action = self.safety_layer.override_action(action, distances)
                
                # Execute action
                self._execute_action(final_action)
                
                # Performance tracking
                loop_time = time.time() - loop_start
                self.loop_times.append(loop_time)
                
                # Sleep to maintain loop rate (1-2 Hz for main loop)
                sleep_time = max(0.5 - loop_time, 0.05)
                time.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in control loop: {e}", exc_info=True)
                self.hardware.stop()
                time.sleep(1)
    
    def _handle_obstacle_avoid(self, distances: dict) -> str:
        """Handle OBSTACLE_AVOID state"""
        # Let safety layer decide
        is_safe, _ = self.safety_layer.check_safety()
        
        if is_safe:
            # Return to previous state
            if self.previous_state == RobotState.TARGET_VISIBLE:
                self._transition_to_state(RobotState.TARGET_VISIBLE, "Obstacle cleared, resuming goal seeking")
            else:
                self._transition_to_state(RobotState.SEARCH_ROTATE, "Obstacle cleared, searching")
            return "stop"
        
        # Emergency maneuver
        if distances['back'] and distances['back'] > 0.3:
            return "backward"
        elif distances['left'] and distances['right']:
            if distances['left'] > distances['right']:
                return "turn_left"
            else:
                return "turn_right"
        else:
            return "stop"
    
    def _handle_target_visible(self, distances: dict) -> str:
        """Handle TARGET_VISIBLE state - uses Q-learning (LAYER 3)"""
        # Get VLA detections
        frame = self.vla.get_frame()
        if frame is None:
            self._transition_to_state(RobotState.SEARCH_ROTATE, "No camera frame")
            return "stop"
        
        # Detect objects
        objects = self.vla.detector.detect_objects(frame)
        
        # Find target
        target = None
        for obj in objects:
            if obj.name.lower() == self.target_class.lower():
                target = obj
                break
        
        if target and target.confidence > CONFIG.VLA_CONFIDENCE_THRESH:
            # Target visible - use Q-learning
            self.target_lost_counter = 0
            action, reward = self.goal_seeking_layer.decide_goal_seeking_action(target, distances)
            logger.debug(f"Q-Learning: action={action}, reward={reward:.2f}")
            return action
        
        else:
            # Target lost
            self.target_lost_counter += 1
            if self.target_lost_counter >= self.target_lost_threshold:
                self._transition_to_state(RobotState.SEARCH_ROTATE, "Target lost")
            return "stop"
    
    def _handle_search_rotate(self, distances: dict) -> str:
        """Handle SEARCH_ROTATE state - rotate to reacquire target"""
        # Get VLA detections
        frame = self.vla.get_frame()
        if frame is None:
            return "stop"
        
        objects = self.vla.detector.detect_objects(frame)
        
        # Check if target reacquired
        for obj in objects:
            if obj.name.lower() == self.target_class.lower():
                if obj.confidence > CONFIG.VLA_CONFIDENCE_THRESH:
                    self._transition_to_state(RobotState.TARGET_VISIBLE, "Target reacquired")
                    return "stop"
        
        # Continue rotating
        self.search_rotation_counter += 1
        
        if self.search_rotation_counter >= self.search_rotation_limit:
            # Full 360° rotation, target not found
            self._transition_to_state(RobotState.EXPLORATION, "Target not in room")
            self.search_rotation_counter = 0
            return "stop"
        
        return "turn_right"  # Rotate 30° increments
    
    def _handle_exploration(self, distances: dict) -> str:
        """Handle EXPLORATION state - uses classical navigation (LAYER 2)"""
        # Periodically check for target
        if self.exploration_layer.exploration_steps % 10 == 0:
            frame = self.vla.get_frame()
            if frame:
                objects = self.vla.detector.detect_objects(frame)
                for obj in objects:
                    if obj.name.lower() == self.target_class.lower():
                        if obj.confidence > CONFIG.VLA_CONFIDENCE_THRESH:
                            self._transition_to_state(RobotState.TARGET_VISIBLE, "Target found during exploration")
                            return "stop"
        
        # Use exploration layer
        action = self.exploration_layer.decide_exploration_action(distances)
        return action
    
    def _transition_to_state(self, new_state: RobotState, reason: str):
        """Transition to a new state"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            
            transition = {
                'timestamp': datetime.now().isoformat(),
                'from': self.previous_state.value if self.previous_state else "INIT",
                'to': new_state.value,
                'reason': reason
            }
            self.state_transition_log.append(transition)
            
            logger.info(f"STATE TRANSITION: {transition['from']} → {transition['to']} ({reason})")
    
    def _execute_action(self, action: str):
        """Execute motor action"""
        if action == "stop":
            self.hardware.stop()
        elif action == "forward":
            self.hardware.move_forward(speed=50, duration=0.5)
        elif action == "backward":
            self.hardware.move_backward(speed=50, duration=0.5)
        elif action == "turn_left":
            self.hardware.turn_left(speed=40, duration=0.3)
        elif action == "turn_right":
            self.hardware.turn_right(speed=40, duration=0.3)
        else:
            self.hardware.stop()
    
    def get_status(self) -> dict:
        """Get current system status"""
        avg_loop_time = sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0
        
        return {
            'state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'safety_enabled': self.safety_layer.enabled,
            'unsafe': self.safety_layer.unsafe,
            'exploration_mode': self.exploration_layer.mode.value,
            'q_table_size': len(self.goal_seeking_layer.agent.q_table),
            'epsilon': self.goal_seeking_layer.agent.epsilon,
            'episode_reward': self.goal_seeking_layer.episode_reward,
            'episode_steps': self.goal_seeking_layer.episode_steps,
            'avg_loop_time': avg_loop_time,
            'target_class': self.target_class,
            'running': self.running
        }

# =================================================================
# MAIN ENTRY POINT
# =================================================================

def main():
    """Main entry point"""
    controller = HierarchicalRLController()
    
    try:
        controller.start()
        
        logger.info("=" * 60)
        logger.info("HIERARCHICAL RL CONTROLLER RUNNING")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Keep main thread alive
        while controller.running:
            time.sleep(5)
            
            # Print status every 5 seconds
            status = controller.get_status()
            logger.info(f"Status - State: {status['state']}, "
                       f"Q-table: {status['q_table_size']} states, "
                       f"Epsilon: {status['epsilon']:.3f}, "
                       f"Reward: {status['episode_reward']:.1f}")
    
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        controller.stop()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
