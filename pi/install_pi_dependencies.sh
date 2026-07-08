#!/bin/bash
# Install Dependencies for Hierarchical RL System on Raspberry Pi
# Run with: bash install_pi_dependencies.sh

echo "============================================================"
echo "Installing Dependencies for Hierarchical RL System"
echo "============================================================"

# Update package list
echo ""
echo "1. Updating package list..."
sudo apt-get update

# Install system dependencies
echo ""
echo "2. Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev python3-numpy python3-opencv

# Install Python packages
echo ""
echo "3. Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy>=1.21.0
pip3 install opencv-python>=4.5.0
pip3 install requests>=2.25.0
pip3 install configparser
pip3 install pillow>=8.0.0

# Verify installations
echo ""
echo "4. Verifying installations..."
python3 -c "import numpy; print('✓ numpy', numpy.__version__)"
python3 -c "import cv2; print('✓ opencv', cv2.__version__)"
python3 -c "import requests; print('✓ requests', requests.__version__)"
python3 -c "import RPi.GPIO as GPIO; print('✓ RPi.GPIO')" 2>/dev/null || echo "⚠ RPi.GPIO not found (install with: sudo apt-get install python3-rpi.gpio)"

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Copy VLA and model1.py files from EDAI folder"
echo "2. Run: sudo python3 hierarchical_rl_main.py"
echo ""
