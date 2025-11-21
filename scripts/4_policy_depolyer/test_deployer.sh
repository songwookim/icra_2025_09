#!/bin/bash
# Quick test script for stiffness policy deployer

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Stiffness Policy Deployer Test Script ===${NC}\n"

# Find latest artifact directory
UNIFIED_DIR="/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts"
if [ -d "$UNIFIED_DIR" ]; then
    LATEST_ARTIFACT=$(ls -t "$UNIFIED_DIR" | head -n 1)
    ARTIFACT_PATH="$UNIFIED_DIR/$LATEST_ARTIFACT"
    echo -e "${GREEN}Found latest artifact:${NC} $ARTIFACT_PATH"
else
    echo -e "${RED}Artifact directory not found!${NC}"
    echo "Please train a model first or specify artifact path manually."
    exit 1
fi

# Check if manifest exists
if [ ! -f "$ARTIFACT_PATH/manifest.json" ]; then
    echo -e "${RED}manifest.json not found in artifact directory!${NC}"
    exit 1
fi

echo -e "${GREEN}Artifact directory validated${NC}\n"

# Build the package
echo -e "${YELLOW}Building hri_falcon_robot_bridge package...${NC}"
cd /home/songwoo/ros2_ws/icra2025
source /opt/ros/humble/setup.bash
colcon build --packages-select hri_falcon_robot_bridge

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}\n"

# Source the workspace
source install/setup.bash

# Choose model type
echo -e "${YELLOW}Choose model type to deploy:${NC}"
echo "1) BC (fastest, smoothest)"
echo "2) Diffusion DDPM (most accurate, slower)"
echo "3) Diffusion DDIM (fast & accurate)"
echo "4) LSTM-GMM (temporal)"
read -p "Enter choice [1-4] (default: 3): " CHOICE

case "$CHOICE" in
    1)
        MODEL_TYPE="bc"
        RATE_HZ="200.0"
        ;;
    2)
        MODEL_TYPE="diffusion"
        SAMPLER="ddpm"
        RATE_HZ="50.0"
        ;;
    4)
        MODEL_TYPE="lstm_gmm"
        RATE_HZ="100.0"
        ;;
    *)
        MODEL_TYPE="diffusion"
        SAMPLER="ddim"
        RATE_HZ="100.0"
        ;;
esac

echo -e "\n${GREEN}Launching deployer with:${NC}"
echo "  Model: $MODEL_TYPE"
echo "  Rate: $RATE_HZ Hz"
if [ "$MODEL_TYPE" = "diffusion" ]; then
    echo "  Sampler: $SAMPLER"
fi
echo "  Artifact: $ARTIFACT_PATH"
echo ""

# Launch the deployer
if [ "$MODEL_TYPE" = "diffusion" ]; then
    ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
        model_type:=$MODEL_TYPE \
        artifact_dir:=$ARTIFACT_PATH \
        rate_hz:=$RATE_HZ \
        diffusion_sampler:=$SAMPLER \
        diffusion_n_samples:=1
else
    ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
        model_type:=$MODEL_TYPE \
        artifact_dir:=$ARTIFACT_PATH \
        rate_hz:=$RATE_HZ
fi
