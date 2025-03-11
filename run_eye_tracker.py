#!/usr/bin/env python3
"""
Eye Tracking Analysis System Launcher

This script launches the eye tracking application.
"""

import os
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module
from src.main import main

if __name__ == "__main__":
    # Launch the application
    main() 