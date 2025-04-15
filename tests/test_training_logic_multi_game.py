import pytest
from unittest.mock import Mock, patch
from criticalnets.training.logic.multi_game import MultiGameLogic

class TestMultiGameLogic:
    def test_init(self):
        """Test initialization with default parameters"""
        logic = MultiGameLogic()
        assert logic.switch_interval == 10
        assert len(logic.game_metrics) == 0

    def test_switch_game(self):
        """Test game switching logic"""
        logic = MultiGameLogic()
        env = Mock()
        env.available_games = ['game1', 'game2']
        env.current_game = 'game1'
        env.make.return_value = 'new_env'
        
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = 'game2'
            new_env = logic._switch_game(env)
            
            mock_choice.assert_called_once_with(['game1', 'game2'])
            env.close.assert_called_once()
            env.make.assert_called_once_with('game2')
            assert new_env == 'new_env'

    def test_run_episode_no_switch(self):
        """Test episode without game switch"""
        logic = MultiGameLogic(switch_interval=10)
        env = Mock()
        env.current_game = 'game1'
        
        reward, metrics = logic.run_episode(env, Mock(), Mock(), 5)
        
        assert reward == 0.0
        assert metrics['game'] == 'game1'
        assert not env.close.called

    def test_run_episode_with_switch(self):
        """Test episode with game switch"""
        logic = MultiGameLogic(switch_interval=10)
        env = Mock()
        env.available_games = ['game1', 'game2']
        env.current_game = 'game1'
        env.make.return_value = env
        
        with patch.object(logic, '_switch_game') as mock_switch:
            mock_switch.return_value = env
            logic.run_episode(env, Mock(), Mock(), 10)
            
            mock_switch.assert_called_once_with(env)

    def test_on_checkpoint(self):
        """Test checkpoint callback saves metrics"""
        logic = MultiGameLogic()
        logic.game_metrics['game1'] = [1.0, 2.0, 3.0]
        logic.game_metrics['game2'] = [4.0, 5.0]
        
        with patch('builtins.print') as mock_print:
            logic.on_checkpoint(100)
            
            assert mock_print.call_count == 3
            mock_print.assert_any_call("Checkpoint at episode 100")
            mock_print.assert_any_call("game1: 2.00 avg reward")
            mock_print.assert_any_call("game2: 4.50 avg reward")
