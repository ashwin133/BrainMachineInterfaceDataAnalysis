"""
calculates the number of times someone can press the space button in a defined time
"""

import keyboard
import time

def count_space_presses(duration):
    count = 0
    end_time = time.time() + duration

    print(f"Press the space bar as many times as you can in {duration} seconds!")

    while time.time() < end_time:                        
        if keyboard.read_key() == 'space':
            count += 1
            # wait until space is released to avoid counting the same press multiple times
            while keyboard.is_pressed('space'):
                pass

    print(f"Time's up! You pressed the space bar {count} times.")

# Set the duration for the count in seconds
duration = 3  # For example, 10 seconds
count_space_presses(duration)
      