import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from criticalnets.training.multi_game_trainer import MultiGameTrainer
from criticalnets.training.logic.base import TrainingLogic

class TestMultiGameTrainer:
    @patch('criticalnets.training.multi_game_trainer.AtariManager')
    @patch('criticalnets.training.multi_game_trainer.KeyboardController')
    def test_init(self, mock_keyboard, mock_atari):
        """Test trainer initialization"""
        config = {
            'memory_size': 50000,
            'render': False,
            'save_dir': 'test_checkpoints'
        }
        mock_logic = Mock(spec=TrainingLogic)
        # Create separate mock instances for policy and target nets
        mock_agent_cls = Mock()
        mock_agent_cls.side_effect = [Mock(), Mock()]
        
        # Setup mocks
        mock_atari.return_value.get_max_action_space.return_value = 18
        mock_atari.return_value.working_games = ['Pong', 'Breakout']
        
        trainer = MultiGameTrainer(config, mock_logic, mock_agent_cls)
        
        # Verify initialization
        assert trainer.config == config
        assert trainer.logic == mock_logic
        assert trainer.max_actions == 18
        mock_atari.assert_called_once()
        mock_keyboard.assert_called_once()
        
        # Verify agent initialization
        mock_agent_cls.assert_called_with(config, 18)
        assert trainer.policy_net == trainer.agent
        assert trainer.target_net != trainer.agent
        assert trainer.optimizer == mock_logic.configure_optimizer.return_value
        assert trainer.memory.memory.maxlen == 50000

    @patch('criticalnets.training.multi_game_trainer.AtariManager')
    @patch('criticalnets.training.multi_game_trainer.KeyboardController')
    def test_make_env(self, mock_keyboard, mock_atari):
        """Test environment creation"""
        config = {'render': True}
        mock_logic = Mock()
        mock_agent_cls = Mock()
        
        with patch('gym.make') as mock_gym:
            mock_gym.return_value = 'env'
            trainer = MultiGameTrainer(config, mock_logic, mock_agent_cls)
            env = trainer._make_env('Pong')
            
            mock_gym.assert_called_once_with(
                'Pong',
                render_mode='human'
            )
            assert env == 'env'

    @patch('criticalnets.training.multi_game_trainer.AtariManager')
    @patch('criticalnets.training.multi_game_trainer.KeyboardController')
    def test_switch_game(self, mock_keyboard, mock_atari):
        """Test game switching logic"""
        config = {}
        mock_logic = Mock()
        mock_agent_cls = Mock()
        mock_atari.return_value.working_games = ['Pong', 'Breakout']
        
        trainer = MultiGameTrainer(config, mock_logic, mock_agent_cls)
        
        with patch.object(trainer, '_make_env') as mock_make_env:
            mock_make_env.return_value = 'new_env'
            mock_env = Mock()
            mock_env.close = Mock()
            
            new_game, new_env = trainer._switch_game('Pong', mock_env)
            
            mock_env.close.assert_called_once()
            mock_make_env.assert_called_once()
            assert new_game in ['Breakout']  # Should switch to the other game
            assert new_env == 'new_env'

    @patch('criticalnets.training.multi_game_trainer.AtariManager')
    @patch('criticalnets.training.multi_game_trainer.KeyboardController')
    @patch('torch.save')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_save_checkpoint(self, mock_makedirs, mock_exists, mock_save, mock_keyboard, mock_atari):
        """Test checkpoint saving"""
        config = {'save_dir': 'test_checkpoints'}
        mock_logic = Mock()
        mock_agent_cls = Mock()
        mock_atari.return_value.working_games = ['Pong']
        
        trainer = MultiGameTrainer(config, mock_logic, mock_agent_cls)
        mock_exists.return_value = False  # Simulate directory doesn't exist
        
        trainer.save_checkpoint('Pong', 100)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with('test_checkpoints')
        
        # Verify save was called with correct arguments
        mock_save.assert_called_once()
        
        # Get all positional arguments
        args, _ = mock_save.call_args
        
        # First arg should be the data dict
        assert isinstance(args[0], dict)
        assert 'state_dict' in args[0]
        assert 'games' in args[0]
        
        # Second arg should be the path string
        assert isinstance(args[1], str)
        assert 'test_checkpoints' in args[1]
        assert 'Pong_100.pt' in args[1]
