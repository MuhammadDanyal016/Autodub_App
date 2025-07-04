"""
Google Colab runner for Enhanced AutoDub API
Use this to easily start the API in Colab
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    packages = [
        "fastapi",
        "uvicorn[standard]", 
        "python-multipart"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                     capture_output=True, text=True)
    
    print("✅ Dependencies installed")

def setup_directories():
    """Setup necessary directories"""
    dirs = ["/tmp/autodub", "/tmp/autodub/output"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

def install_cloudflared():
    """Install cloudflared for tunnel support"""
    try:
        # Check if already installed
        result = subprocess.run(['cloudflared', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ cloudflared already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("🔧 Installing cloudflared...")
    commands = [
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
        "sudo dpkg -i cloudflared-linux-amd64.deb || true",
        "rm -f cloudflared-linux-amd64.deb"
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)
    
    # Verify installation
    try:
        result = subprocess.run(['cloudflared', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ cloudflared installed successfully")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️ cloudflared installation failed, but API will still work without tunnel")
    return False

def monitor_tunnel():
    """Monitor tunnel health and restart if needed"""
    import requests
    import time
    
    # Wait for API to start up
    print("⏳ Waiting for API to start...")
    time.sleep(10)
    
    while True:
        try:
            # Check if API is responding
            response = requests.get('http://localhost:8000/tunnel/health', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if not data.get('tunnel_healthy', False):
                    print("⚠️ Tunnel unhealthy, API will attempt auto-recovery")
                elif data.get('tunnel_active', False):
                    tunnel_url = data.get('tunnel_url')
                    if tunnel_url:
                        print(f"✅ Tunnel active: {tunnel_url}")
            else:
                print(f"⚠️ API health check failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            # API not ready yet, skip this check
            pass
        except Exception as e:
            print(f"⚠️ Tunnel monitoring error: {e}")
        
        time.sleep(60)  # Check every minute

def run_api():
    """Run the Enhanced AutoDub API with monitoring"""
    print("🎬 Starting Enhanced AutoDub API v4.0...")
    
    # Set environment variables
    os.environ["API_HOST"] = "0.0.0.0"
    os.environ["API_PORT"] = "8000"
    
    # Start tunnel monitoring in background
    monitor_thread = threading.Thread(target=monitor_tunnel, daemon=True)
    monitor_thread.start()
    
    # Import and run the API
    try:
        import uvicorn
        from enhanced_autodub_api import app
        
        print("🚀 API starting on http://0.0.0.0:8000")
        print("📚 Documentation: http://0.0.0.0:8000/docs")
        print("🔍 Tunnel Health: http://0.0.0.0:8000/tunnel/health")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False

def main():
    """Main function to setup and run the API"""
    print("🎬 Enhanced AutoDub API v4.0 - Google Colab Setup")
    print("=" * 50)
    
    # Setup steps
    setup_directories()
    install_dependencies()
    install_cloudflared()
    
    print("\n" + "=" * 50)
    print("🚀 Setup complete! Starting API...")
    print("=" * 50)
    
    # Run the API
    run_api()

if __name__ == "__main__":
    main()
