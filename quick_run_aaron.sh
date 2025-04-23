#!/bin/bash

sudo apt install python-is-python3 && mkdir logs
pip install -r requirements.txt && pip install -e .
nohup python scripts/train_atari.py --config criticalnets/config/a2c_sweep.yaml > a2c_script.log 2>&1 &
nohup python scripts/train_atari.py --config criticalnets/config/a2c_crit_sweep.yaml > a2c_crit_script.log 2>&1 &