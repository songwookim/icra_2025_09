# hri_falcon_robot_bridge (ROS 2 Humble)

Pipeline:
```
[Force Sensor Node (py)]
  publishes per-sensor /force_sensor/s{i}/wrench (geometry_msgs/WrenchStamped)
  publishes combined  /force_sensor/wrench_array (std_msgs/Float64MultiArray, shape: [num_sensors, 6])
      ↓
[Falcon Node (C++)]
  subscribes /force_sensor/wrench_array → apply forces (array-only; legacy /wrench is ignored)
  publishes /falcon/encoders (std_msgs/Int32MultiArray)
  publishes /falcon/position (geometry_msgs/Vector3Stamped)
  on startup: optional initial posture PD to target encoders
      ↓
[Robot Controller Node (py)]
  subscribes encoders/position → map → Dynamixel set_joint_positions(...)
```

```
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer 
sudo vi /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   # change to 16 -> 1
   
```
## Build
```bash
cd ~/humble_ws/src
# unzip and place this folder here
unzip /path/to/hri_falcon_robot_bridge.zip -d .
cd ..
colcon build --packages-select hri_falcon_robot_bridge
source install/setup.bash
```

## Run
```bash
ros2 launch hri_falcon_robot_bridge bridge.launch.py
```
Default parameters use mock data for the force sensor so you can test the wiring first. To pass extra parameters at launch time:

```bash
ros2 launch hri_falcon_robot_bridge bridge.launch.py \
  falcon_node.init_posture_enable:=true \
  falcon_node.init_enc_target:="[-500,-500,-500]" \
  falcon_node.force_sensor_index:=0
```

## Integrate your real controllers
- Force sensor (Python):
  - Set `use_mock:=False` in `force_sensor_node.py` parameters and ensure your controller driver is importable.
  - Each sensor must provide 6D wrench `[fx, fy, fz, tx, ty, tz]`.
  - Node publishes per-sensor `WrenchStamped` and a combined `Float64MultiArray` with layout dims `[sensor, axis]` (axis length = 6).
- Falcon device (C++):
  - `falcon_node.cpp` uses libnifalcon directly: firmware load, homing, I/O loop, `getEncoderValues()`, `setForces(...)`.
  - Startup can drive to an initial encoder posture via a PD routine (configurable; defaults to `[-500,-500,-500]`).
- Dynamixel controller:
  - `robot_controller_node.py` expects a `DynamixelControl`-compatible API (`connect`, `set_joint_positions`).

## Topics
- `/force_sensor/wrench_array` : `std_msgs/Float64MultiArray` with shape `[num_sensors, 6]`
- `/force_sensor/s{i}/wrench` : `geometry_msgs/WrenchStamped` per sensor (i = 1..N)
- `/force_sensor/wrench` : `geometry_msgs/WrenchStamped` (legacy, first sensor)
- `/falcon/encoders` : `std_msgs/Int32MultiArray` (len=3)
- `/falcon/position` : `geometry_msgs/Vector3Stamped`

## Parameters (tunable)
- falcon_node (C++):
  - `force_scale` (double, default 1500.0): multiply incoming values to device force units
  - `publish_rate_hz` (double, default 200.0): encoder/position publish rate
  - `frame_id` (string, default `falcon_base`)
  - `falcon_id` (int, default 0): select device index
  - `force_sensor_index` (int, default 0): select starting row from `wrench_array` when mapping to forces
  - Initial posture (encoder-space PD on startup):
    - `init_posture_enable` (bool, default true)
    - `init_enc_target` (int[3], default [-500,-500,-500])
    - `init_kp` (double, default 100.0), `init_kd` (double, default 0.1)
    - `init_force_limit` (int, default 1000)
    - `init_max_loops` (int, default 20000)
    - `init_stable_eps` (int, default 5), `init_stable_count` (int, default 0 → no stability wait)
- force_sensor_node (Python):
  - `publish_rate_hz` (double): publish frequency
  - `use_mock` (bool): enable mock generator
  - `config_path` (string): optional config
  - `num_sensors` (int): number of sensors (rows in `wrench_array`)
- robot_controller_node (Python):
  - `ids` (int[3]), `mode` (int)
  - `scale[3]`, `offset[3]`, `clip_min[3]`, `clip_max[3]`
  - `use_encoders` (bool): True→subscribe `/falcon/encoders`, False→`/falcon/position`

## Notes
- The launch file defaults to `use_mock=True` so you can verify ROS wiring without hardware.
- Switch to real hardware by setting `use_mock:=False` and providing device-specific controllers.
- `falcon_node` applies forces only from `/force_sensor/wrench_array` (legacy `/wrench` is subscribed but ignored).
- Initial posture PD runs after homing; disable via `init_posture_enable:=false` if undesired.
