#!/usr/bin/env python3
"""
Setup script for Quran Semantic Analysis project.
This script helps install dependencies and set up the environment.
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_java():
    """Check if Java is installed."""
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        print("✓ Java is installed")
        return True
    except FileNotFoundError:
        print("✗ Java is not installed")
        return False

def install_java_macos():
    """Install Java on macOS using Homebrew."""
    print("Installing Java on macOS...")
    commands = [
        ("brew install openjdk@11", "Install OpenJDK 11"),
        ("echo 'export PATH=\"/opt/homebrew/opt/openjdk@11/bin:$PATH\"' >> ~/.zshrc", "Add Java to PATH"),
        ("source ~/.zshrc", "Reload shell configuration")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_java_ubuntu():
    """Install Java on Ubuntu/Debian."""
    print("Installing Java on Ubuntu/Debian...")
    commands = [
        ("sudo apt-get update", "Update package list"),
        ("sudo apt-get install -y openjdk-11-jdk", "Install OpenJDK 11"),
        ("sudo update-alternatives --config java", "Configure Java")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def download_jqurantree():
    """Download JQuranTree JAR file."""
    jar_path = "jqurantree.jar"
    if os.path.exists(jar_path):
        print(f"✓ JQuranTree JAR already exists at {jar_path}")
        return True
    
    print("Downloading JQuranTree JAR file...")
    try:
        # Try to download from GitHub releases
        url = "https://github.com/jqurantree/jqurantree/releases/latest/download/jqurantree.jar"
        urllib.request.urlretrieve(url, jar_path)
        print(f"✓ JQuranTree JAR downloaded to {jar_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download JQuranTree JAR: {e}")
        print("Please manually download JQuranTree JAR from: https://github.com/jqurantree/jqurantree")
        return False

def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Install basic dependencies
    if not run_command("pip install -r requirements.txt", "Install Python packages"):
        return False
    
    # Install system dependencies for Arabic text processing
    system = platform.system().lower()
    if system == "linux":
        commands = [
            ("sudo apt-get update", "Update package list"),
            ("sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev", "Install system dependencies")
        ]
    elif system == "darwin":  # macOS
        commands = [
            ("brew install libxml2 libxslt", "Install system dependencies")
        ]
    else:
        print("⚠️  Please install system dependencies manually for your OS")
        return True
    
    for command, description in commands:
        run_command(command, description)  # Don't fail if system deps fail
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["results", "data", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Quran Semantic Analysis - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Check Java
    if not check_java():
        system = platform.system().lower()
        if system == "darwin":
            install_java_macos()
        elif system == "linux":
            install_java_ubuntu()
        else:
            print("⚠️  Please install Java manually for your operating system")
    
    # Download JQuranTree
    download_jqurantree()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("✗ Failed to install Python dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python quran_analysis.py")
    print("2. Check the 'results/' directory for output files")
    print("3. View 'quran_relationship_matrix.png' for visualization")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
