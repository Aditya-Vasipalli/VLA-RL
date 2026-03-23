# VLA+RL ROBOT CONFIGURATION

# =================================================================
# NETWORK SETTINGS
# =================================================================

# VLA Server Configuration
VLA_SERVER_IP = "192.168.1.100"  # CHANGE THIS to your laptop IP
VLA_SERVER_PORT = 9999
WEB_INTERFACE_PORT = 5000

# =================================================================
# ROBOT PARAMETERS
# =================================================================

# Default target object
DEFAULT_TARGET = "bottle"

# Movement parameters
FORWARD_SPEED = 60      # PWM duty cycle (0-100)
TURN_SPEED = 70         # PWM duty cycle (0-100) 
ACTION_DURATION = 0.5   # seconds per action

# Sensor thresholds
OBSTACLE_THRESHOLD = 0.15  # meters (15cm)
MAX_SENSOR_RANGE = 2.0     # meters

# =================================================================
# LEARNING PARAMETERS  
# =================================================================

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

# Episode settings
MAX_STEPS_PER_EPISODE = 1000
MAX_TIME_PER_EPISODE = 300  # seconds (5 minutes)
EPISODES_BETWEEN_SAVES = 10

# =================================================================
# VLA TIMING
# =================================================================

# When to query VLA (seconds)
MAX_TIME_BETWEEN_VLA = 15    # Force VLA query after this time
EXPLORATION_VLA_INTERVAL = 10  # Query VLA in exploration mode
SEARCH_VLA_INTERVAL = 5      # Query VLA when searching for lost target

# VLA optimization
OPTIMAL_IMAGE_WIDTH = 320
OPTIMAL_IMAGE_HEIGHT = 240
OPTIMAL_JPEG_QUALITY = 50

# =================================================================
# REWARD SYSTEM
# =================================================================

REWARDS = {
    'step_penalty': -0.1,
    'collision_penalty': -10,
    'target_found_bonus': 2,
    'target_centered_bonus': 5,
    'target_visible_bonus': 1,
    'distance_improvement_bonus': 3,
    'distance_worsening_penalty': -1,
    'mission_success_bonus': 50
}

# =================================================================
# HARDWARE PINS
# =================================================================

# Motor driver pins (TB6612FNG)
MOTOR_PINS = {
    'AI1': 19, 'AI2': 16, 'PWMA': 13,  # Left motors
    'BI1': 20, 'BI2': 21, 'PWMB': 12,  # Right motors
    'STBY': 18  # Standby
}

# Ultrasonic sensor pins (HC-SR04)  
SENSOR_PINS = {
    'FRONT': {'trig': 26, 'echo': 27},
    'LEFT':  {'trig': 24, 'echo': 25}, 
    'RIGHT': {'trig': 5,  'echo': 6},
    'BACK':  {'trig': 22, 'echo': 23}
}

# Camera settings
CAMERA_INDEX = 0
CAMERA_WARMUP_TIME = 2  # seconds