#!/bin/bash

TARGET_DIR="/workspace/multimodal-navigation-transformer/test/src"

if [ "$(pwd)" != "$TARGET_DIR" ]; then
    cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
fi

# Continue with the rest of your script
echo "Now in : $(pwd)"

session_name="vilint_evaluation_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v  # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v  # split it into two halves

tmux select-pane -t 0
tmux send-keys "source /opt/ros/jazzy/setup.bash" Enter
tmux send-keys "conda activate dune-pttf" Enter
tmux send-keys "python3 /workspace/multimodal-navigation-transformer/deployment/src/deploy_vilint.py --imgwaypoints $@" Enter

tmux select-pane -t 1
tmux send-keys "source /opt/ros/jazzy/setup.bash" Enter
tmux send-keys "conda activate dune-pttf" Enter
tmux send-keys "python3 /workspace/multimodal-navigation-transformer/deployment/src/pd_controller.py" Enter

tmux select-pane -t 2
tmux send-keys "conda activate isaac" Enter
tmux send-keys "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/isaac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib" Enter
tmux send-keys "python3 evaluate.py --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error" Enter

tmux -2 attach-session -t $session_name