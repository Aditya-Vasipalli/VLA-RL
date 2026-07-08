# Windows Server Setup Script
# Run this in PowerShell as Administrator

Write-Host "🖥️  Setting up YOLO Server for RL Robot" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`n1️⃣  Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found! Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n2️⃣  Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  venv already exists, skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n3️⃣  Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host "`n4️⃣  Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Check GPU
Write-Host "`n5️⃣  Checking GPU availability..." -ForegroundColor Yellow
$gpuCheck = python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>&1
Write-Host $gpuCheck

# Get IP address
Write-Host "`n6️⃣  Your laptop IP addresses:" -ForegroundColor Yellow
$ipAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*"} | Select-Object IPAddress, InterfaceAlias
$ipAddresses | Format-Table -AutoSize
Write-Host "👆 Use one of these IPs in Pi's config.ini" -ForegroundColor Cyan

# Download YOLO model (if not exists)
Write-Host "`n7️⃣  Checking YOLO model..." -ForegroundColor Yellow
if (Test-Path "yolov8s.pt") {
    Write-Host "✅ yolov8s.pt already exists" -ForegroundColor Green
} else {
    Write-Host "📦 Model will auto-download on first run" -ForegroundColor Yellow
}

# Firewall rule
Write-Host "`n8️⃣  Firewall configuration..." -ForegroundColor Yellow
Write-Host "⚠️  You may need to allow port 8000 in Windows Firewall" -ForegroundColor Yellow
Write-Host "   Run: netsh advfirewall firewall add rule name='YOLO Server' dir=in action=allow protocol=TCP localport=8000" -ForegroundColor Gray

# Final instructions
Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Copy one of the IP addresses above" -ForegroundColor White
Write-Host "   2. Update Pi's config.ini with: http://YOUR_IP:8000" -ForegroundColor White
Write-Host "   3. Start server: python yolo_server.py" -ForegroundColor White
Write-Host "   4. Test endpoint: http://YOUR_IP:8000/health" -ForegroundColor White
Write-Host "`n🚀 Ready to launch!" -ForegroundColor Green
