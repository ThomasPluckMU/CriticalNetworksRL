from typing import Dict, Type
import torch
import torch.nn as nn
from importlib import import_module
from pathlib import Path

class BaseAtariAgent(nn.Module):
    """Base class that all Atari agents must implement"""
    def __init__(self, config: Dict, action_space: int):
        super().__init__()
        self.config = config
        self.action_space = action_space
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Agents must implement forward()")

# Auto-discover and register agents
AGENT_REGISTRY = {}

agents_dir = Path(__file__).parent
for file in agents_dir.glob("*.py"):
    if file.name.startswith("_"):
        continue
    module = import_module(f"criticalnets.agents.{file.stem}")
    for name, cls in vars(module).items():
        if isinstance(cls, type) and issubclass(cls, BaseAtariAgent) and cls != BaseAtariAgent:
            AGENT_REGISTRY[name] = cls

def get_agent_class(name: str) -> Type[BaseAtariAgent]:
    """Get agent class by name from registry"""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available agents: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name]
