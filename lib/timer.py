"""Timer"""

import sys
import os
import time


class Timer:
    """Timer"""

    def __init__(self, target=None):
        """Initialize timer
        :rtype: object
        """
        self.start_time = None
        self.end_time = None
        if target is None:
            self.target = os.path.basename(sys.argv[0])

    def start(self):
        """Starts timer"""
        self.start_time = time.time()

    def end(self):
        """Print time elapsed since execution began"""
        self.end_time = time.time()
        hrs = (self.end_time - self.start_time) // 3600
        minutes = ((self.end_time - self.start_time) % 3600) // 60
        sec = (self.end_time - self.start_time) % 60
        print(f"{self.target} "
              f"- Execution time: "
              f"{int(hrs)} hour(s) {int(minutes)} minute(s) {sec:.3f} second(s).")
