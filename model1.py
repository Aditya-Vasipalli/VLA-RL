# RL Project Skeleton for Robot Navigation
# -----------------------------------------
# This file contains the main structure for Q-learning-based robot navigation.
# It is heavily commented for team clarity and future extension.

import numpy as np
import random

# --- State Definition ---
# State consists of:
# - Ultrasonic sensor readings (distances in 4 directions)
# - VLA output: pixel offset of goal center from camera center
# - Raw encoder data (wheel ticks, rotations)
#
# Example state: (front_dist, left_dist, right_dist, back_dist, goal_offset_px, encoder_left, encoder_right)

# --- Action Definition ---
# Actions are defined by direct control of wheel encoders:
# - Set left/right wheel speeds (e.g., [-1, 0, 1] for backward, stop, forward)
# - Stop
#
# Example actions: [('left_speed', 'right_speed')], e.g., (1, 1) for forward, (1, -1) for turn, (0, 0) for stop

# For simplicity, define a small set of possible speed combinations:
ACTIONS = [
    (1, 1),    # Move forward
    (-1, -1),  # Move backward
    (1, -1),   # Turn right
    (-1, 1),   # Turn left
    (0, 0)     # Stop
]

# --- Q-table Initialization ---
# For simplicity, we use a dict. In practice, state space may be large, so consider discretization.
Q_table = {}

# --- RL Hyperparameters ---
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate

# --- HITL (Human-in-the-Loop) Parameters ---
HITL_THRESHOLD = 5  # Number of rounds without goal before triggering HITL
hitl_required = False
no_goal_counter = 0

# --- Reward Function ---
# Rewards:
# +100: robot is 1cm in front of goal
# +10: goal is centered in camera view
# +1: moving towards goal (goal offset decreases)
# -50: losing sight of goal
# -100: collision or too close to obstacle (<1cm)

def get_reward(state, prev_state, goal_found, collision):
    """
    Calculate reward based on state transitions and sensor data.
    """
    reward = 0
    # Big reward for reaching goal
    if goal_found and state['front_dist'] <= 1:
        reward += 100
    # Smaller reward for centering goal
    if goal_found and abs(state['goal_offset_px']) < 5:
        reward += 10
    # Small reward for moving towards goal
    if goal_found and abs(state['goal_offset_px']) < abs(prev_state['goal_offset_px']):
        reward += 1
    # Punishment for losing sight of goal
    if not goal_found:
        reward -= 50
    # Punishment for collision or being too close
    if collision or min(state['front_dist'], state['left_dist'], state['right_dist'], state['back_dist']) < 1:
        reward -= 100
    return reward

# --- HITL Placeholder Function ---
def await_human_input():
    """
    Placeholder for HITL API integration.
    When called, pause RL and wait for human input via API/web interface.
    Resume RL after receiving input.
    """
    print("HITL required: waiting for human intervention...")
    # TODO: Integrate with web API to receive human input
    # Example: poll an endpoint, wait for new goal/action
    pass

# --- Main RL Loop ---
def rl_main_loop():
    """
    Main loop for Q-learning RL agent.
    Handles state updates, action selection, reward calculation, and HITL logic.
    """
    global no_goal_counter, hitl_required
    state = get_initial_state()  # Implement sensor reading here
    prev_state = state.copy()
    while True:
        # --- Check for HITL ---
        if hitl_required:
            await_human_input()
            hitl_required = False
            no_goal_counter = 0
        # --- Action Selection (epsilon-greedy) ---
        state_key = state_to_key(state)
        if random.random() < EPSILON or state_key not in Q_table:
            action = random.choice(ACTIONS)
        else:
            action = max(Q_table[state_key], key=Q_table[state_key].get)
        # --- Execute Action ---
        next_state, goal_found, collision = execute_action(action, state)
        # --- Reward Calculation ---
        reward = get_reward(next_state, state, goal_found, collision)
        # --- Q-table Update ---
        next_state_key = state_to_key(next_state)
        if state_key not in Q_table:
            Q_table[state_key] = {a: 0 for a in ACTIONS}
        if next_state_key not in Q_table:
            Q_table[next_state_key] = {a: 0 for a in ACTIONS}
        Q_table[state_key][action] += ALPHA * (reward + GAMMA * max(Q_table[next_state_key].values()) - Q_table[state_key][action])
        # --- HITL Trigger Logic ---
        if not goal_found:
            no_goal_counter += 1
            if no_goal_counter >= HITL_THRESHOLD:
                hitl_required = True
        else:
            no_goal_counter = 0
        # --- Prepare for next step ---
        prev_state = state
        state = next_state
        # TODO: Add logging, ROS integration, and sensor reading functions

# --- Helper Functions ---
def get_initial_state():
    """
    Read sensors and return initial state dict.
    Replace with actual sensor reading code.
    """
    return {
        'front_dist': 10,
        'left_dist': 10,
        'right_dist': 10,
        'back_dist': 10,
        'goal_offset_px': 0,
        'encoder_left': 0,
        'encoder_right': 0
    }

def state_to_key(state):
    """
    Convert state dict to a tuple key for Q-table.
    Discretize values if needed.
    """
    return (
        int(state['front_dist']),
        int(state['left_dist']),
        int(state['right_dist']),
        int(state['back_dist']),
        int(state['goal_offset_px']),
        int(state['encoder_left']),
        int(state['encoder_right'])
    )

def execute_action(action, state):
    """
    Execute the chosen action and return (next_state, goal_found, collision).
    Replace with actual robot control code.
    """
    # TODO: Integrate with ROS and hardware
    next_state = state.copy()  # Simulate next state
    goal_found = True  # Simulate goal detection
    collision = False  # Simulate collision detection
    return next_state, goal_found, collision

# --- Entry Point ---
if __name__ == "__main__":
    rl_main_loop()
