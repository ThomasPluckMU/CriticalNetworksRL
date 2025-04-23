#!/bin/bash

sudo apt install python-is-python3 && mkdir logs
pip install -r requirements.txt && python ./setup.py
nohup python scripts/train_atari.py --config criticalnets/config/ppo_sweep.yaml > ppo_script.log 2>&1 &
nohup python scripts/train_atari.py --config criticalnets/config/dqn_td_sweep.yaml > dqn_td_script.log 2>&1 &