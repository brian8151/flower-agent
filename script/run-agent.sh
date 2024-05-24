#!/bin/bash

# Assignments of arguments
NODE=$1
CMD=${2:-start}  # Default to 'start' if no second argument is given

# Configuration
APP_DIR_BASE="/home/ubuntu"
APP_DIR="$APP_DIR_BASE/flower-agent"
LOG_PATH="$APP_DIR_BASE"  # Modified path for clarity and to ensure it exists
PYTHON_BIN="/usr/local/bin/python3.10"

cd $APP_DIR
echo "Running from fl-agent directory: $(pwd)"

# Function to install dependencies if needed
install_deps() {
    if [ ! -f ".deps_installed" ]; then
        echo "Installing dependencies from requirements.txt..."
        $PYTHON_BIN -m pip install -r requirements.txt
        touch .deps_installed
        echo "Dependencies installed."
    else
        echo "Dependencies already installed."
    fi
}

# Function to start the application
start_app() {
    install_deps
    echo "Starting Flower agent application..."
      cd $APP_DIR_BASE
    nohup $PYTHON_BIN flower-agent --node-id $NODE 2>&1 &
    echo "Flower agent started in the background, logs: $LOG_PATH/flwr-agent-$NODE.log"
}

# Function to stop the application
stop_app() {
    local pid=$(ps aux | grep -i 'flower-agent' | grep -v grep | awk '{print $2}')
    if [[ ! -z "$pid" ]]; then
        echo "Found process $pid, shutting down fl-agent..."
        kill $pid
        sleep 2
        if ps -p $pid > /dev/null; then
            echo "Process $pid did not terminate, forcing shutdown..."
            kill -9 $pid
        fi
        echo "FL agent stopped."
    else
        echo "No process found for fl-agent."
    fi
}

# Handle the command line argument
case "$CMD" in
    start)
        stop_app  # Ensure the application is not already running
        start_app
        ;;
    stop)
        stop_app
        ;;
    *)
        echo "Usage: $0 <node-id> {start|stop}"
        exit 1
        ;;
esac
