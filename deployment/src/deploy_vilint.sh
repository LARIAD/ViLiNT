#!/bin/bash

TARGET_DIR="/workspace/multimodal-navigation-transformer/deployment/src"

if [ "$(pwd)" != "$TARGET_DIR" ]; then
    cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
fi

# Continue with the rest of your script
echo "Now in : $(pwd)"

session_name="vilint_deployment_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

tmux select-pane -t 0
tmux send-keys "source /opt/ros/humble/setup.bash" Enter
tmux send-keys "python3 deploy_vilint.py --imgwaypoints $@" Enter

tmux select-pane -t 1
tmux send-keys "source /opt/ros/humble/setup.bash" Enter
tmux send-keys "python3 pd_controller.py" Enter

tmux select-pane -t 2
tmux send-keys "source /opt/ros/humble/setup.bash" Enter
tmux send-keys "ros2 run rviz2 rviz2" Enter

tmux select-pane -t 3
tmux send-keys "source /opt/ros/humble/setup.bash" Enter
tmux send-keys "ros2 bag record -o $session_name /odom /point_cloud /rgb" 

tmux -2 attach-session -t $session_name