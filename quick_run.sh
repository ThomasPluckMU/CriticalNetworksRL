#!/bin/bash

# Setup
sudo apt install python-is-python3 && mkdir -p logs
pip install -r requirements.txt && pip install -e .

# Create a single monitoring script that will run detached from your terminal
cat > monitor_jobs.sh << 'EOF'
#!/bin/bash

# Start the first PPO job and save its PID
echo "Starting ppo_agent_sweep_0.yaml..."
python scripts/train_atari.py --config criticalnets/configs/ppo_agent_sweep_0.yaml > logs/ppo_sweep_0.log 2>&1 &
PPO_PID_0=$!

# Start the second PPO job and save its PID  
echo "Starting ppo_agent_sweep_1.yaml..."
python scripts/train_atari.py --config criticalnets/configs/ppo_agent_sweep_1.yaml > logs/ppo_sweep_1.log 2>&1 &
PPO_PID_1=$!

# Function to monitor a process and start the next job when it completes
monitor_and_run_next() {
    local pid=$1
    local config=$2
    local log_file=$3
    
    echo "Monitoring PID $pid, will start $config when complete" >> logs/monitor.log
    
    # Wait for the process to complete
    while kill -0 $pid 2>/dev/null; do
        sleep 30  # Check every 30 seconds
    done
    
    # Process is done, start the next job
    echo "Process $pid completed. Starting $config..." >> logs/monitor.log
    python scripts/train_atari.py --config "$config" > "$log_file" 2>&1
}

# Run the monitors sequentially to ensure they complete regardless of shell session
monitor_and_run_next $PPO_PID_0 "criticalnets/configs/dqn_td_sweep_0.yaml" "logs/dqn_td_sweep_0.log"
monitor_and_run_next $PPO_PID_1 "criticalnets/configs/dqn_td_sweep_1.yaml" "logs/dqn_td_sweep_1.log"

echo "All jobs completed successfully" >> logs/monitor.log
EOF

# Make the monitoring script executable
chmod +x monitor_jobs.sh

# Launch the monitoring script with nohup so it's completely detached from your terminal
nohup ./monitor_jobs.sh > logs/main_process.log 2>&1 &

echo "Monitoring process started. You can safely close this terminal."
echo "Check logs/monitor.log for progress updates."