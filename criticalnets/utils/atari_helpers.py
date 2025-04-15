import numpy as np
import torch
import time
import os
import signal
import sys
import threading
try:
    import msvcrt  # Windows
except ImportError:
    import keyboard  # Linux/macOS
import random
from typing import Tuple
from collections import deque, namedtuple

Transition = namedtuple('Transition', 
    ('state', 'action', 'next_state', 'reward', 'done', 'game_id'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale and crop/downsample"""
    frame = frame.mean(axis=2)  # Convert to grayscale
    frame = frame[34:194, 8:152]  # Crop
    frame = frame[::2, ::2]  # Downsample
    return frame.astype(np.float32) / 255.0

class KeyboardController:
    """Handles keyboard controls for rendering speed"""
    def __init__(self, initial_delay=0.01):
        self.render_delay = initial_delay
        self.fast_mode = False
        self._running = True
        
    def start(self):
        """Start keyboard control thread"""
        print("\nKeyboard controls:")
        print("  + : Increase render delay (slower)")
        print("  - : Decrease render delay (faster)")
        print("  0 : Reset to default delay")
        print("  f : Toggle super fast mode (no delay)")
        
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread
        
    def _run(self):
        while self._running:
            key = None
            try:
                if msvcrt.kbhit():  # Windows
                    key = msvcrt.getch()
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore')
            except NameError:
                if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    key = '+'
                elif keyboard.is_pressed('-'):
                    key = '-'
                elif keyboard.is_pressed('0'):
                    key = '0'
                elif keyboard.is_pressed('f'):
                    key = 'f'
                if key == '+':
                    self.render_delay += 0.005
                    print(f"Render delay: {self.render_delay:.3f}s")
                elif key == '-':
                    self.render_delay = max(0, self.render_delay - 0.005)
                    print(f"Render delay: {self.render_delay:.3f}s")
                elif key == '0':
                    self.render_delay = 0.01
                    print(f"Render delay reset to default: {self.render_delay:.3f}s")
                elif key == 'f':
                    self.fast_mode = not self.fast_mode
                    self.render_delay = 0 if self.fast_mode else 0.01
                    print(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
            time.sleep(0.1)
            
    def stop(self):
        self._running = False
