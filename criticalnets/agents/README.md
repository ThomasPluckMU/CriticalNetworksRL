# Agents Module

## 1. What is this module?
This module contains implementations of Atari game agents that inherit from `BaseAtariAgent`. It provides:
- A base class (`BaseAtariAgent`) that defines the interface all agents must implement
- Automatic discovery and registration of agent implementations
- A registry system to look up agent classes by name

## 2. How to use it
```python
from criticalnets.agents import get_agent_class

# Get an agent class by name
AgentClass = get_agent_class('GatedAtariUDQN')

# Create an agent instance
agent = AgentClass(config, action_space)

# Use the agent
q_values = agent(state_tensor)
```

## 3. How to add new agents
To add a new agent:
1. Create a new Python file in this directory
2. Implement a class that inherits from `BaseAtariAgent`
3. Implement the required `forward()` method
4. The agent will be automatically discovered and registered

Example minimal agent:
```python
from . import BaseAtariAgent
import torch.nn as nn

class MyNewAgent(BaseAtariAgent):
    def __init__(self, config, action_space):
        super().__init__(config, action_space)
        # Define your network layers here
        
    def forward(self, x):
        # Implement forward pass
        return q_values
