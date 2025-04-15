import pytest
from unittest.mock import Mock, patch
import numpy as np
from criticalnets.utils.atari_helpers import *

class TestAtariHelpers:
    def test_preprocess_frame(self):
        # Test with simple input frame
        frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        processed = preprocess_frame(frame)
        assert processed.shape == (80, 72)  # Matches actual implementation
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1.0

    def test_replay_memory(self):
        memory = ReplayMemory(10)
        transition = (np.zeros(1), 1, np.zeros(1), 1.0, False, 0)
        
        # Test push
        memory.push(*transition)
        assert len(memory) == 1
        
        # Test sample
        with patch('random.sample', return_value=[transition]) as mock_sample:
            batch = memory.sample(1)
            assert len(batch) == 1
            mock_sample.assert_called_once()

    @patch('msvcrt.getch')
    @patch('msvcrt.kbhit')
    def test_keyboard_controller(self, mock_kbhit, mock_getch):
        controller = KeyboardController()
        
        # Test key controls
        mock_kbhit.return_value = True
        mock_getch.return_value = '+'
        thread = controller.start()
        
        # Verify delay increases
        initial_delay = controller.render_delay
        mock_getch.return_value = '+'  # Send increase delay command
        time.sleep(0.2)  # Let thread process
        assert controller.render_delay == initial_delay + 0.005  # Exact expected increase
        
        controller.stop()
        thread.join()
