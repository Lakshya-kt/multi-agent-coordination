"""
conftest.py — pytest configuration for multi-agent-coordination tests.
Adds the project root to sys.path so all imports resolve correctly.
"""
import sys
import os

# Ensure project root is on the path regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
