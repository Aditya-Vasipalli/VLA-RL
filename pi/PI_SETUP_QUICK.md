# Quick Start Guide - Raspberry Pi Setup

## Current Error: Missing numpy
```
ModuleNotFoundError: No module named 'numpy'
```

## Solution: Install Dependencies on Pi

### Option 1: Quick Install (Recommended)
```bash
# On the Pi, run:
pip3 install numpy opencv-python requests configparser pillow
```

### Option 2: Use Requirements File
```bash
# Copy requirements.txt to Pi first
scp requirements.txt arora@raspberrypi:~/

# Then on Pi:
pip3 install -r requirements.txt
```

### Option 3: System Packages (Faster on Pi)
```bash
# These are pre-compiled for ARM, install faster:
sudo apt-get update
sudo apt-get install -y python3-numpy python3-opencv python3-pip
pip3 install requests configparser pillow
```

## After Installing Dependencies

### You'll also need VLA and model1.py files:
```bash
# On Windows, from EDAI_final directory:
scp EDAI/vla_mvp/vla.py arora@raspberrypi:~/
scp EDAI/vla_mvp/utils.py arora@raspberrypi:~/
scp EDAI/model1.py arora@raspberrypi:~/
```

### Then test:
```bash
# On Pi:
sudo python3 hierarchical_rl_main.py
```

## Expected Output (After Dependencies Installed)
```
============================================================
HIERARCHICAL RL CONTROLLER - INITIALIZING
============================================================
Safety Layer initialized (Layer 1)
Exploration Layer initialized (Layer 2)
Goal Seeking Layer initialized (Layer 3 - Q-Learning)
...
```

## Common Issues

### 1. "No module named 'numpy'"
**Fix:** `pip3 install numpy`

### 2. "No module named 'cv2'" (OpenCV)
**Fix:** `pip3 install opencv-python` OR `sudo apt-get install python3-opencv`

### 3. "No module named 'requests'"
**Fix:** `pip3 install requests`

### 4. "No module named 'RPi.GPIO'"
**Fix:** `sudo apt-get install python3-rpi.gpio`

### 5. Import errors from vla.py or model1.py
**Fix:** Copy those files from EDAI folder to Pi (see commands above)

## Quick Command Sequence

Run these on **Raspberry Pi**:
```bash
# 1. Install dependencies
pip3 install numpy opencv-python requests configparser pillow

# 2. Verify installation
python3 -c "import numpy, cv2, requests; print('✓ All packages installed')"

# 3. Run the system (after copying VLA/model1 files)
sudo python3 hierarchical_rl_main.py
```

## Files Needed on Pi

Minimum required files:
- ✅ hierarchical_rl_main.py (already copied)
- ✅ hardware.py (already copied)
- ✅ config.ini (already copied)
- ⚠️ vla.py (from EDAI/vla_mvp/)
- ⚠️ utils.py (from EDAI/vla_mvp/)
- ⚠️ model1.py (from EDAI/)

## Next Step Commands

**Right now on your Pi terminal, run:**
```bash
pip3 install numpy opencv-python requests configparser pillow
```

This should take 2-5 minutes to install all packages.
