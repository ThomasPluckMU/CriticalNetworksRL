import yaml
from pathlib import Path
from typing import Dict, Any, List
import itertools
from copy import deepcopy

from ..agents import get_agent_class
from ..training.logic import get_logic_class

import hashlib, json


class ConfigHandler:
    """
    Handles loading and validation of configuration files for hyperparameter tuning
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
    def config_to_hex(self, config):
        # Filter out non-serializable objects
        config_copy = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        return hashlib.md5(json.dumps(config_copy, sort_keys=True).encode()).hexdigest()[:8]

    def generate_configs(self) -> List[Dict[str, Any]]:
        config_list = []

        with open(self.config_path, "r") as file:
            yaml_dict = yaml.safe_load(file)

            global_agents = yaml_dict["agents"]

            remaining_dict = {
                key: value
                for key, value in yaml_dict.items()
                if key not in ["hyperparameters", "agents", "logics"]
            }

            for ak, av in global_agents.items():
                # Check that agent has logics defined
                if "logics" not in av or not av["logics"]:
                    raise ValueError(
                        f"Agent '{ak}' must have at least one logic defined"
                    )

                # Handle case where agent_params might be missing
                agent_params = av.get("agent_params", {})

                for lk, lv in av["logics"].items():
                    # Handle missing logic_params
                    logic_params = lv.get("logic_params", {})

                    # Convert agent params to list format if they're not already
                    agent_param_keys = list(agent_params.keys())
                    agent_param_values = []

                    # If there are no agent params, we still need one combination (empty)
                    if not agent_param_keys:
                        agent_param_combos = [()]
                    else:
                        for apk in agent_param_keys:
                            apv = agent_params[apk]
                            if not isinstance(apv, list):
                                agent_params[apk] = [apv]
                            agent_param_values.append(agent_params[apk])

                        agent_param_combos = list(
                            itertools.product(*agent_param_values)
                        )

                    # Convert logic params to list format if they're not already
                    logic_param_keys = list(logic_params.keys())
                    logic_param_values = []

                    # If there are no logic params, we still need one combination (empty)
                    if not logic_param_keys:
                        logic_param_combos = [()]
                    else:
                        for lpk in logic_param_keys:
                            lpv = logic_params[lpk]
                            if not isinstance(lpv, list):
                                logic_params[lpk] = [lpv]
                            logic_param_values.append(logic_params[lpk])

                        logic_param_combos = list(
                            itertools.product(*logic_param_values)
                        )

                    # Combine agent and logic params
                    for agent_combo in agent_param_combos:
                        for logic_combo in logic_param_combos:
                            # Create a new config dict
                            config = deepcopy(remaining_dict)

                            # Add agent and logic identifiers
                            config["agent"] = ak
                            config["logic"] = lk

                            # Add agent param values
                            for i, apk in enumerate(agent_param_keys):
                                config[apk] = agent_combo[i]

                            # Add logic param values
                            for i, lpk in enumerate(logic_param_keys):
                                config[lpk] = logic_combo[i]

                            config['run_id'] = self.config_to_hex(config)

                            config_list.append(config)

        return config_list
