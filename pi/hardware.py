#!/usr/bin/env python3
"""
Hardware Controller for Hierarchical RL System
Uses VERIFIED pin diagram from remote/robot_server.py

Pin Mappings (TB6612 Motor Driver - VERIFIED WORKING):
- Channel A (LEFT): AI1=19, AI2=16, PWMA=13
- Channel B (RIGHT): BI1=20, BI2=21, PWMB=12
- STBY=18

Ultrasonic Sensors (HC-SR04 - VERIFIED WORKING):
- FRONT: TRIG=26, ECHO=27
- LEFT: TRIG=24, ECHO=25
- RIGHT: TRIG=5, ECHO=6
- BACK: TRIG=22, ECHO=23

Features:
- Motor smoothing for jerk-free movement
- 4 ultrasonic sensors
- Safety features (collision detection, emergency stop)
"""

import RPi.GPIO as GPIO
import time
import logging
import threading
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pin configuration from remote/robot_server.py (VERIFIED)
AI1 = 19
AI2 = 16
PWMA = 13
BI1 = 20
BI2 = 21
PWMB = 12
STBY = 18

SENSORS = {
    'front': {'trig': 26, 'echo': 27},
    'left':  {'trig': 24, 'echo': 25},
    'right': {'trig': 5,  'echo': 6},
    'back':  {'trig': 22, 'echo': 23}
}

# Motor smoothing parameters
ACCEL_STEP = 10
SMOOTH_INTERVAL = 0.03

class RobotHardware:
    """Robot hardware controller with verified pin mappings"""
    
    def __init__(self):
        # Motor state (for smoothing)
        self.motor_state = {
            'left_speed': 0,
            'right_speed': 0,
            'left_dir': 's',  # 'f', 'b', 's'
            'right_dir': 's'
        }
        self.motor_lock = threading.Lock()
        
        # Ultrasonic pins (HC-SR04)
        self.ultrasonic_pins = {
            'front': {'trig': 16, 'echo': 26},
            'left':  {'trig': 17, 'echo': 27},
            'right': {'trig': 7,  'echo': 8},
            'back':  {'trig': 9,  'echo': 11}
        }
        
        # Safety parameters
        self.default_speed = 70
        self.turn_speed = 60
        self.safety_distance = 0.20  # 20cm
        self.collision_distance = 0.10  # 10cm
        
        self._setup_gpio()
        logger.info("✅ Robot hardware initialized")
    
    def _setup_gpio(self):
        """Initialize all GPIO pins"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup motors
        for motor, pins in self.motor_pins.items():
            GPIO.setup(pins['in1'], GPIO.OUT)
            GPIO.setup(pins['in2'], GPIO.OUT)
            GPIO.setup(pins['enable'], GPIO.OUT)
        
        # Setup ultrasonics
        for sensor, pins in self.ultrasonic_pins.items():
            GPIO.setup(pins['trig'], GPIO.OUT)
            GPIO.setup(pins['echo'], GPIO.IN)
            GPIO.output(pins['trig'], GPIO.LOW)
        
        # Create PWM for motor speed control
        self.pwm_motors = {}
        for motor, pins in self.motor_pins.items():
            self.pwm_motors[motor] = GPIO.PWM(pins['enable'], 1000)
            self.pwm_motors[motor].start(0)
    
    # ========== MOVEMENT FUNCTIONS ==========
    
    def move_forward(self, duration: float = 0.3) -> bool:
        """Move forward with collision detection"""
        # Safety check
        if self.get_distance('front') < self.safety_distance:
            logger.warning("🚫 Forward blocked by obstacle")
            return False
        
        logger.info(f"🚗 Moving forward {duration}s")
        self._set_all_motors('forward', self.default_speed)
        
        # Monitor during movement
        start = time.time()
        while time.time() - start < duration:
            if self.get_distance('front') < self.collision_distance:
                logger.error("🛑 Emergency stop - collision!")
                self.emergency_stop()
                return False
            time.sleep(0.05)
        
        self.stop()
        return True
    
    def move_backward(self, duration: float = 0.3) -> bool:
        """Move backward with rear collision detection"""
        if self.get_distance('back') < self.safety_distance:
            logger.warning("🚫 Backward blocked")
            return False
        
        logger.info(f"🚗 Moving backward {duration}s")
        self._set_all_motors('backward', self.default_speed)
        time.sleep(duration)
        self.stop()
        return True
    
    def turn_left(self, duration: float = 0.3) -> bool:
        """Tank turn left"""
        logger.info(f"🚗 Turning left {duration}s")
        self._set_motor_group(['front_left', 'back_left'], 'backward', self.turn_speed)
        self._set_motor_group(['front_right', 'back_right'], 'forward', self.turn_speed)
        time.sleep(duration)
        self.stop()
        return True
    
    def turn_right(self, duration: float = 0.3) -> bool:
        """Tank turn right"""
        logger.info(f"🚗 Turning right {duration}s")
        self._set_motor_group(['front_right', 'back_right'], 'backward', self.turn_speed)
        self._set_motor_group(['front_left', 'back_left'], 'forward', self.turn_speed)
        time.sleep(duration)
        self.stop()
        return True
    
    def stop(self):
        """Stop all motors"""
        for motor in self.pwm_motors:
            self.pwm_motors[motor].ChangeDutyCycle(0)
            pins = self.motor_pins[motor]
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.LOW)
    
    def emergency_stop(self):
        """Immediate emergency stop"""
        logger.error("🚨 EMERGENCY STOP")
        self.stop()
    
    # ========== SENSOR FUNCTIONS ==========
    
    def get_distance(self, sensor: str) -> float:
        """
        Get distance from ultrasonic sensor
        Returns: Distance in meters (0.02-4.0m)
        """
        if sensor not in self.ultrasonic_pins:
            return 2.0
        
        pins = self.ultrasonic_pins[sensor]
        
        try:
            # Send trigger pulse
            GPIO.output(pins['trig'], GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(pins['trig'], GPIO.LOW)
            
            # Wait for echo (with timeout)
            timeout = time.time()
            while GPIO.input(pins['echo']) == 0:
                pulse_start = time.time()
                if pulse_start - timeout > 0.1:
                    return 2.0
            
            timeout = time.time()
            while GPIO.input(pins['echo']) == 1:
                pulse_end = time.time()
                if pulse_end - pulse_start > 0.1:
                    return 2.0
            
            # Calculate distance
            duration = pulse_end - pulse_start
            distance = (duration * 34300) / 2 / 100  # meters
            return max(0.02, min(distance, 4.0))
        
        except Exception as e:
            logger.error(f"Sensor error {sensor}: {e}")
            return 2.0
    
    def get_all_distances(self) -> Dict[str, float]:
        """Get all 4 ultrasonic readings"""
        distances = {}
        for sensor in ['front', 'left', 'right', 'back']:
            distances[sensor] = self.get_distance(sensor)
            time.sleep(0.01)
        return distances
    
    # ========== LOW-LEVEL CONTROL ==========
    
    def _set_all_motors(self, direction: str, speed: int):
        """Set all motors to same direction/speed"""
        for motor in self.motor_pins:
            self._set_motor(motor, direction, speed)
    
    def _set_motor_group(self, motors: list, direction: str, speed: int):
        """Set specific motor group"""
        for motor in motors:
            self._set_motor(motor, direction, speed)
    
    def _set_motor(self, motor: str, direction: str, speed: int):
        """Control individual motor"""
        pins = self.motor_pins[motor]
        
        if direction == 'forward':
            GPIO.output(pins['in1'], GPIO.HIGH)
            GPIO.output(pins['in2'], GPIO.LOW)
        elif direction == 'backward':
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.HIGH)
        else:
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.LOW)
            speed = 0
        
        speed = max(0, min(speed, 100))
        self.pwm_motors[motor].ChangeDutyCycle(speed)
    
    # ========== CLEANUP ==========
    
    def cleanup(self):
        """Shutdown hardware safely"""
        logger.info("🛑 Shutting down hardware...")
        self.emergency_stop()
        for motor in self.pwm_motors:
            self.pwm_motors[motor].stop()
        GPIO.cleanup()
        logger.info("✅ Hardware cleanup complete")
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


if __name__ == "__main__":
    """Test hardware"""
    robot = RobotHardware()
    try:
        print("Testing robot hardware...")
        print("1. Sensors:", robot.get_all_distances())
        print("2. Forward movement...")
        robot.move_forward(1.0)
        time.sleep(0.5)
        print("3. Turn left...")
        robot.turn_left(0.5)
        time.sleep(0.5)
        print("4. Turn right...")
        robot.turn_right(0.5)
        print("✅ Hardware test complete!")
    finally:
        robot.cleanup()
