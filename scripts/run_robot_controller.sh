#!/usr/bin/env bash
set -euo pipefail
# Source ROS 2 and workspace overlays
source /opt/ros/humble/setup.bash
if [ -f "$(dirname "$0")/../install/setup.bash" ]; then
  source "$(dirname "$0")/../install/setup.bash"
fi
exec /usr/bin/python3 -u $(dirname "$0")/../src/hri_falcon_robot_bridge/hri_falcon_robot_bridge/robot_controller_node.py "$@"
