#!/bin/bash
# Raspberry Pi Setup Script

echo "🤖 Setting up RL Robot on Raspberry Pi"
echo "======================================="

# Check Python version
echo ""
echo "1️⃣  Checking Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "✅ $python_version"
else
    echo "❌ Python 3 not found! Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Update system
echo ""
echo "2️⃣  Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-venv

# Install dependencies
echo ""
echo "3️⃣  Installing Python dependencies..."
pip3 install -r requirements.txt --user

# Check camera
echo ""
echo "4️⃣  Checking camera..."
if [ -c /dev/video0 ]; then
    echo "✅ Camera detected at /dev/video0"
else
    echo "⚠️  No camera detected. Check USB connection."
fi

# Check GPIO permissions
echo ""
echo "5️⃣  Checking GPIO permissions..."
if groups | grep -q 'gpio'; then
    echo "✅ User in GPIO group"
else
    echo "⚠️  Adding user to GPIO group..."
    sudo usermod -a -G gpio $USER
    echo "✅ Added. Please reboot: sudo reboot"
fi

# Test imports
echo ""
echo "6️⃣  Testing Python imports..."
python3 -c "import RPi.GPIO; import cv2; import numpy; import requests; print('✅ All imports successful')" 2>&1

# Configuration reminder
echo ""
echo "📝 Next steps:"
echo "   1. Edit config.ini with your server IP"
echo "   2. Test hardware: python3 hardware.py"
echo "   3. Start robot: python3 main.py"
echo ""
echo "✅ Setup complete!"
