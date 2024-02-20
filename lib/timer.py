"""
This module contains the Timer class.

The Timer class is used to measure the execution time of code.
It includes methods for starting, stopping, and resetting the timer,
as well as getting the elapsed time.

Typical usage example:

    >>> timer = Timer()
    >>> timer.start()
    >>> # Some code here...
    >>> timer.end()
"""
import sys
import os
import time


class Timer:
    """
    Timer class for measuring the execution time of code.

    Methods:
        __init__(self, target=None): Initializes the timer.
        start(self): Starts the timer.
        end(self): Prints the time elapsed since execution began.
    """

    def __init__(self, target=None):
        """Initialize timer with optional target name.

            Args:
                target (str, optional): The name of the target. Defaults to None.    
        """

        self.start_time = None
        self.end_time = None
        if target is None:
            self.target = os.path.basename(sys.argv[0])
        else:
            self.target = target

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
