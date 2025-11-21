# Devcontainer Dockerfile for hri_falcon_robot_bridge
# Base: ROS 2 Humble (minimal ros-base) + build tools + Python ML libs
FROM ros:humble-ros-base

# Non-interactive
ARG DEBIAN_FRONTEND=noninteractive

# System packages (build tools, visualization, mujoco prerequisites)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget unzip \
    python3-pip python3-venv python3-colcon-common-extensions \
    ros-humble-rqt* ros-humble-tf2* \
    libegl1 libgl1-mesa-glx libosmesa6 libglfw3 libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages (pin lightweight where sensible)
RUN pip3 install --no-cache-dir \
    numpy scipy scikit-learn pandas matplotlib seaborn \
    torch --index-url https://download.pytorch.org/whl/cpu \
    mujoco==3.1.2 mujoco-python==3.1.2 \
    # Optional machine learning extras
    diffusers einops \
    && python3 -c "import mujoco" || echo 'Mujoco import failed (ensure license if required)'

# Create workspace directory
WORKDIR /workspace

# ROS env setup
SHELL ["/bin/bash", "-lc"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Default command
CMD ["bash"]
