#!/bin/bash
set -e
source /opt/ros/humble/setup.bash

# Initialize rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  rosdep init || true
fi
rosdep update || true

# Install dependencies via rosdep (ignore src if mounted elsewhere)
if [ -d src ]; then
  rosdep install --from-paths src --ignore-src -r -y || true
fi

# Create colcon workspace structure if not present
mkdir -p /workspace/src || true

# Build if package present
if ls /workspace/src 1> /dev/null 2>&1; then
  colcon build --symlink-install || true
fi

echo "[post_create] Devcontainer setup complete."