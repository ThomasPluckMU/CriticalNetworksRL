import pytest
from unittest.mock import Mock, patch
import numpy as np
from criticalnets.environments.atari_manager import AtariManager

class TestAtariManager:
    @patch.object(AtariManager, '_verify_games')
    def test_init(self, mock_verify):
        """Test AtariManager initialization"""
        mock_verify.return_value = ['ALE/Pong-v5', 'ALE/Breakout-v5']
        manager = AtariManager()
        
        mock_verify.assert_called_once()
        assert manager.working_games == ['ALE/Pong-v5', 'ALE/Breakout-v5']

    @patch('gymnasium.make')
    def test_get_max_action_space(self, mock_make):
        """Test max action space calculation"""
        # Setup mock environments
        mock_pong = Mock()
        mock_pong.action_space.n = 6
        mock_breakout = Mock()
        mock_breakout.action_space.n = 4
        
        # Make mock_make return different envs for different games
        def make_side_effect(game):
            if game == 'ALE/Pong-v5':
                return mock_pong
            return mock_breakout
        mock_make.side_effect = make_side_effect
        
        # Initialize with mock working games
        manager = AtariManager()
        manager.working_games = ['ALE/Pong-v5', 'ALE/Breakout-v5']
        
        assert manager.get_max_action_space() == 6

    @patch.object(AtariManager, '_verify_games')
    @patch('gymnasium.make')
    def test_get_max_action_space_fallback(self, mock_make, mock_verify):
        """Test fallback when no games work"""
        mock_verify.return_value = ['ALE/Pong-v5']  # Pretend we have one working game
        mock_make.side_effect = Exception("Game failed")
        
        manager = AtariManager()
        assert manager.get_max_action_space() == 18  # Default fallback
