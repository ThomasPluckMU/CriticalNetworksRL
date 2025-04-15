import pytest
from unittest.mock import Mock, patch, MagicMock
from criticalnets.training.single_game_trainer import SingleGameTrainer
from criticalnets.training.logic.base import TrainingLogic
from criticalnets.agents.gated_atari_udqn import GatedAtariUDQN as BaseAtariAgent

class TestSingleGameTrainer:
    @patch('criticalnets.training.single_game_trainer.AtariManager')
    @patch('criticalnets.training.single_game_trainer.KeyboardController')
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
        mock_atari.return_value.working_games = ['Pong']
        
        trainer = SingleGameTrainer(config, mock_logic, mock_agent_cls)
        
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

    @patch('criticalnets.training.single_game_trainer.AtariManager')
    @patch('criticalnets.training.single_game_trainer.KeyboardController')
    def test_make_env(self, mock_keyboard, mock_atari):
        """Test environment creation"""
        config = {'render': True}
        mock_logic = Mock()
        mock_agent_cls = Mock()
        
        with patch('gym.make') as mock_gym:
            mock_gym.return_value = 'env'
            trainer = SingleGameTrainer(config, mock_logic, mock_agent_cls)
            env = trainer._make_env('Pong')
            
            mock_gym.assert_called_once_with(
                'Pong',
                render_mode='human'
            )
            assert env == 'env'

    @patch('criticalnets.training.single_game_trainer.AtariManager')
    @patch('criticalnets.training.single_game_trainer.KeyboardController')
    @patch('torch.save')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_save_checkpoint(self, mock_makedirs, mock_exists, mock_save, mock_keyboard, mock_atari):
        """Test checkpoint saving"""
        config = {'save_dir': 'test_checkpoints'}
        mock_logic = Mock()
        mock_agent_cls = Mock()
        mock_atari.return_value.working_games = ['Pong']
        
        trainer = SingleGameTrainer(config, mock_logic, mock_agent_cls)
        mock_exists.return_value = False  # Simulate directory doesn't exist
        
        trainer.save_checkpoint('Pong', 100)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with('test_checkpoints')
        
        # Verify save was called with correct arguments
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        
        # First arg should be the data dict
        assert isinstance(args[0], dict)
        assert 'state_dict' in args[0]
        assert 'game' in args[0]
        
        # Second arg should be the path string
        assert isinstance(args[1], str)
        assert 'test_checkpoints' in args[1]
        assert 'Pong_single_100.pt' in args[1]
