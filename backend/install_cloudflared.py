"""
Install cloudflared for Cloudflare Tunnel support
Run this script in Google Colab before starting the API
"""

import subprocess
import sys
import os

def install_cloudflared():
    """Install cloudflared binary for Cloudflare Tunnel"""
    try:
        print("🔧 Installing cloudflared...")
        
        # Download and install cloudflared
        commands = [
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
            "sudo dpkg -i cloudflared-linux-amd64.deb",
            "rm cloudflared-linux-amd64.deb"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(f"Error: {result.stderr}")
                return False
        
        # Verify installation
        result = subprocess.run(['cloudflared', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ cloudflared installed successfully: {result.stdout.strip()}")
            return True
        else:
            print("❌ cloudflared installation verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def setup_environment():
    """Setup environment for Google Colab"""
    try:
        # Create necessary directories
        os.makedirs("/tmp/autodub", exist_ok=True)
        os.makedirs("/tmp/autodub/output", exist_ok=True)
        print("✅ Created necessary directories")
        
        # Install required Python packages
        packages = [
            "fastapi",
            "uvicorn[standard]",
            "python-multipart"
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True, text=True)
        
        print("✅ Python packages installed")
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Enhanced AutoDub API for Google Colab")
    
    if setup_environment():
        print("✅ Environment setup complete")
    else:
        print("❌ Environment setup failed")
        sys.exit(1)
    
    if install_cloudflared():
        print("✅ cloudflared installation complete")
        print("\n🎬 You can now run the Enhanced AutoDub API!")
        print("Run: python enhanced_autodub_api.py")
    else:
        print("❌ cloudflared installation failed")
        print("You can still run the API, but Cloudflare Tunnel won't be available")
