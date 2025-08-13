#!/usr/bin/env python3
"""Generate letter A with domain restrictions."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.main import main

if __name__ == "__main__":
    # Override command line args to test letter A
    sys.argv = ["", "A"]
    main()
