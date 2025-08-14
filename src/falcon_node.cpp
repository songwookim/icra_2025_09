#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <csignal>
#include <iostream>
#include <array>
#include <thread>

// ROS2 includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

// libnifalcon includes (conditional)
#include "falcon/core/FalconLogger.h"
#include "falcon/core/FalconDevice.h"
#include "falcon/firmware/FalconFirmwareNovintSDK.h"
#include "falcon/util/FalconFirmwareBinaryNvent.h"
using namespace libnifalcon;

using namespace std::chrono_literals;

/*
 * This node wraps a Novint Falcon via libnifalcon.
 * It subscribes to /force_sensor/wrench and sets Falcon motor forces,
 * and publishes /falcon/encoders (Int32MultiArray) and /falcon/position (Vector3Stamped).
 */
class FalconNode : public rclcpp::Node {
public:
  FalconNode() : Node("falcon_node"), falcon_initialized_(false) {
    // Parameters
    force_scale_ = this->declare_parameter<double>("force_scale", 1500.0); // maps N to Falcon units (int16)
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 200.0);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "falcon_base");
    falcon_id_ = this->declare_parameter<int>("falcon_id", 0); // which falcon to use (0-based)
    force_sensor_index_ = this->declare_parameter<int>("force_sensor_index", 0); // which row from array topic

    // Initial posture parameters
    init_posture_enable_ = this->declare_parameter<bool>("init_posture_enable", true);
    auto init_target = this->declare_parameter<std::vector<int>>("init_enc_target", std::vector<int>{-500, -500, -500});
    if (init_target.size() >= 3) {
      init_target_enc_ = {init_target[0], init_target[1], init_target[2]};
    }
    init_kp_ = this->declare_parameter<double>("init_kp", 100.0);
    init_kd_ = this->declare_parameter<double>("init_kd", 0.1);
    init_force_limit_ = this->declare_parameter<int>("init_force_limit", 1000);
    init_max_loops_ = this->declare_parameter<int>("init_max_loops", 20000);
    init_stable_eps_ = this->declare_parameter<int>("init_stable_eps", 5);
    init_stable_count_req_ = this->declare_parameter<int>("init_stable_count", 0); // 0: don't wait for stability

    // Interfaces
    sub_force_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/force_sensor/wrench", 10,
      std::bind(&FalconNode::on_force, this, std::placeholders::_1));

    // Also support the combined array topic: layout [sensor, axis] (axis = fx,fy,fz,tx,ty,tz)
    sub_force_array_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/force_sensor/wrench_array", 10,
      std::bind(&FalconNode::on_force_array, this, std::placeholders::_1));

    pub_enc_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("/falcon/encoders", 10);
    pub_pos_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("/falcon/position", 10);

    // Device init
    init_device();

    // Timer loop to read device and publish
    if (falcon_initialized_) {
      auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
      timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&FalconNode::on_timer, this));
    }
  }

  ~FalconNode() override {
    if (falcon_initialized_) {
      // Set forces to zero and close device
      if (firmware_) {
        firmware_->setForces({0, 0, 0});
        firmware_->setLEDStatus(0);
        device_.runIOLoop();
      }
      device_.close();
      RCLCPP_INFO(get_logger(), "Falcon device closed");
    }
  }

private:
  // Helper: device readiness (assumes libnifalcon is available)
  bool is_ready() const {
    return falcon_initialized_ && static_cast<bool>(firmware_);
  }

  void init_device() {
    RCLCPP_INFO(get_logger(), "Initializing Falcon device...");
    
    // Set firmware type
    device_.setFalconFirmware<FalconFirmwareNovintSDK>();
    firmware_ = device_.getFalconFirmware();
    
    // Get device count
    unsigned int num_falcons = 0;
    if (!device_.getDeviceCount(num_falcons)) {
      RCLCPP_ERROR(get_logger(), "Cannot get device count");
      return;
    }
    
    RCLCPP_INFO(get_logger(), "Falcons found: %d", (int)num_falcons);
    
    if (num_falcons == 0) {
      RCLCPP_ERROR(get_logger(), "No falcons found, exiting...");
      return;
    }
    
    if (falcon_id_ >= (int)num_falcons) {
      RCLCPP_ERROR(get_logger(), "Requested falcon ID %d, but only %d falcons available", falcon_id_, num_falcons);
      return;
    }
    
    // Open the specified falcon
    RCLCPP_INFO(get_logger(), "Opening falcon %d", falcon_id_);
    if (!device_.open(falcon_id_)) {
      RCLCPP_ERROR(get_logger(), "Cannot open falcon %d", falcon_id_);
      return;
    }
    RCLCPP_INFO(get_logger(), "Opened falcon %d", falcon_id_);
    
    // Load firmware if needed
    if (!device_.isFirmwareLoaded()) {
      RCLCPP_INFO(get_logger(), "Loading firmware...");
      for (int i = 0; i < 10; ++i) {
        if (!firmware_->loadFirmware(true, NOVINT_FALCON_NVENT_FIRMWARE_SIZE, 
                                     const_cast<uint8_t*>(NOVINT_FALCON_NVENT_FIRMWARE))) {
          RCLCPP_WARN(get_logger(), "Could not load firmware, attempt %d", i + 1);
        } else {
          RCLCPP_INFO(get_logger(), "Firmware loaded successfully");
          break;
        }
      }
      
      if (!device_.isFirmwareLoaded()) {
        RCLCPP_ERROR(get_logger(), "Firmware didn't load correctly. Try running again.");
        return;
      }
    }
    
    // Perform homing
    RCLCPP_INFO(get_logger(), "Starting homing procedure...");
    firmware_->setHomingMode(true);
    
    // Wait for homing with proper timeout and status checking
    int homing_timeout = 0;
    const int max_homing_attempts = 5000; // Increase timeout
    
    while (!firmware_->isHomed() && homing_timeout < max_homing_attempts) {
      if (!device_.runIOLoop()) {
        // IO failed, continue trying
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        homing_timeout++;
        continue;
      }
      
      auto status = firmware_->getHomingModeStatus();
      if (homing_timeout % 500 == 0) { // Log every 0.5 seconds
        RCLCPP_INFO(get_logger(), "Homing progress... status: %d, timeout: %d/%d", 
                   status, homing_timeout, max_homing_attempts);
      }
      
      homing_timeout++;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    if (firmware_->isHomed()) {
      RCLCPP_INFO(get_logger(), "Homing completed successfully");
      firmware_->setHomingMode(false);
      
      // Set initial forces and LED (following the test pattern)
      firmware_->setForces({0, 0, 0});
      firmware_->setLEDStatus(1); // Turn on first LED like in the test
      device_.runIOLoop();
      
      falcon_initialized_ = true;
      RCLCPP_INFO(get_logger(), "Falcon device initialized successfully!");

      // Drive to initial posture if enabled
      if (init_posture_enable_) {
        drive_to_initial_posture();
      }
    } else {
      RCLCPP_WARN(get_logger(), "Homing failed after %d attempts - continuing anyway for testing", homing_timeout);
      firmware_->setHomingMode(false);
      
      // Continue even if homing failed for testing purposes
      firmware_->setForces({0, 0, 0});
      firmware_->setLEDStatus(1);
      device_.runIOLoop();
      
  falcon_initialized_ = true;
  RCLCPP_WARN(get_logger(), "Falcon device initialized without proper homing");
    }
  }

  void on_force(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    if (!is_ready()) {
      return;
    }
  // Do not apply forces here; only on_force_array should drive the device.
  (void)msg; // unused
  }

  void on_force_array(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (!is_ready()) {
      return;
    }
    const auto &data = msg->data;
    if (data.size() < 6) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "wrench_array too small: %zu", data.size());
      return;
    }
    // Expect layout [sensor, axis] with axis size 6 (fx,fy,fz,tx,ty,tz)
    const int row_len = 6;
    const int sensors = static_cast<int>(data.size() / row_len);
    if (sensors <= 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "wrench_array sensors computed as 0");
      return;
    }
    int idx = force_sensor_index_;
    if (idx < 0) idx = 0;
    if (idx >= sensors) idx = sensors - 1;
    const size_t off = static_cast<size_t>(idx * row_len);
    double sensor1_sum = data[off + 0] + data[off + 1] + data[off + 2] +
                          data[off + 3] + data[off + 4] + data[off + 5];
    double sensor2_sum = data[off + 6] + data[off + 7] + data[off + 8] +
                          data[off + 9] + data[off + 10] + data[off + 11];
    double sensor3_sum = data[off + 12] + data[off + 13] + data[off + 14] +
                          data[off + 15] + data[off + 16] + data[off + 17];
    // static int counter = 0;
    // if (++counter % 50 == 0) { // Log every 50 messages
    //   RCLCPP_INFO(get_logger(), "\n\n Sensor1 command: [%lf, %lf, %lf, %lf, %lf, %lf] // sum : %lf", data[off + 0], data[off + 1], data[off + 2], data[off + 3], data[off + 4], data[off + 5], sensor1_sum);
    //   RCLCPP_INFO(get_logger(), "Sensor2 command: [%lf, %lf, %lf, %lf, %lf, %lf] // sum : %lf", data[off + 6], data[off + 7], data[off + 8], data[off + 9], data[off + 10], data[off + 11], sensor2_sum);
    //   RCLCPP_INFO(get_logger(), "Sensor3 command: [%lf, %lf, %lf, %lf, %lf, %lf] // sum : %lf \n\n", data[off + 12], data[off + 13], data[off + 14], data[off + 15], data[off + 16], data[off + 17], sensor3_sum);
    // }

    apply_force_xyz(sensor1_sum, sensor2_sum, sensor3_sum);
  }

  void on_timer() {
    if (!is_ready()) {
      return;
    }
    // Run IO loop to communicate with device (matching falcon_test.cpp pattern)
    if (!device_.runIOLoop()) {
      return; // Skip this cycle if communication failed
    }
    
    // Read encoder values (like in falcon_test.cpp)
    auto encoder_values = firmware_->getEncoderValues();
    std::array<int, 3> enc = {encoder_values[0], encoder_values[1], encoder_values[2]};
    
    // Get position (you can implement kinematics here)
    // For now, we'll use a simple mapping from encoders to position
    // This matches the pattern from falcon_test.cpp where encoders are the primary data
    std::array<double, 3> pos = {
      enc[0] * 0.0001,  // Simple scaling - similar to test output format
      enc[1] * 0.0001,
      enc[2] * 0.0001
    };
    
    // Optional: Update LED status periodically (like falcon_test.cpp)
    static int led_counter = 0;
    if (++led_counter % 1000 == 0) {  // Change LED every 5 seconds at 200Hz
      int led_state = (led_counter / 1000) % 4; // Cycle through 4 states
      if (led_state == 0) firmware_->setLEDStatus(0);      // All off
      else if (led_state == 1) firmware_->setLEDStatus(1); // LED 1
      else if (led_state == 2) firmware_->setLEDStatus(2); // LED 2  
      else firmware_->setLEDStatus(4);                     // LED 3
    }


    // Publish encoders
    std_msgs::msg::Int32MultiArray enc_msg;
    enc_msg.data = {enc[0], enc[1], enc[2]};
    pub_enc_->publish(enc_msg);

    // Publish position
    geometry_msgs::msg::Vector3Stamped p;
    p.header.stamp = now();
    p.header.frame_id = frame_id_;
    p.vector.x = pos[0];
    p.vector.y = pos[1];
    p.vector.z = pos[2];
    pub_pos_->publish(p);
    
    // Optional: Log encoder values periodically (like falcon_test.cpp printf)
    static int counter = 0;
    if (++counter % 50 == 0) {  // Log every 5 seconds at 200Hz
      RCLCPP_INFO(get_logger(), "Enc1: %5d | Enc2: %5d | Enc3: %5d | Forces: [%d, %d, %d]", 
                  enc[0], enc[1], enc[2], last_cmd_[0], last_cmd_[1], last_cmd_[2]);
    }
  }

  // Helper: scale, clamp, and send forces to device; update last_cmd_
  void apply_force_xyz(double sensor1_sum, double sensor2_sum, double sensor3_sum) {
    int v0 = static_cast<int>(std::round(sensor1_sum * force_scale_));
    int v1 = static_cast<int>(std::round(sensor2_sum * force_scale_));
    int v2 = static_cast<int>(std::round(sensor3_sum * force_scale_));
    int limit = 2000;
    v0 = std::max(-limit, std::min(limit, v0));
    v1 = std::max(-limit, std::min(limit, v1));
    v2 = std::max(-limit, std::min(limit, v2));

    firmware_->setForces({v0, v1, v2});
    last_cmd_ = {v0, v1, v2};
    static int force_log_counter = 0;
    if (++force_log_counter % 50 == 0) {
      RCLCPP_INFO(get_logger(), "Force command: [%d, %d, %d]", v0, v1, v2);
      RCLCPP_INFO(get_logger(), "Sensor command: [%lf, %lf, %lf] \n\n", sensor1_sum, sensor2_sum, sensor3_sum);
    }
  }

  // Drive Falcon to initial encoder target using a simple PD in encoder space
  void drive_to_initial_posture() {
    if (!is_ready()) return;
    RCLCPP_INFO(get_logger(), "Driving to initial encoders: [%d, %d, %d]",
                init_target_enc_[0], init_target_enc_[1], init_target_enc_[2]);

    std::array<int,3> enc_prev = {0,0,0};
    bool have_prev = false;
    unsigned int loops = 0;
    unsigned int stable_count = 0;
    auto t_prev = std::chrono::steady_clock::now();
    while (rclcpp::ok() && loops < static_cast<unsigned int>(init_max_loops_)) {
      if (!device_.runIOLoop()) continue;

      auto enc = firmware_->getEncoderValues();

      // dt in seconds
      auto t_now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(t_now - t_prev).count();
      if (dt <= 0.0) dt = 1e-3;
      t_prev = t_now;

      // Encoder velocity (ticks/s)
      std::array<double,3> vel = {0.0, 0.0, 0.0};
      if (have_prev) {
        vel[0] = (enc[0] - enc_prev[0]) / dt;
        vel[1] = (enc[1] - enc_prev[1]) / dt;
        vel[2] = (enc[2] - enc_prev[2]) / dt;
      }
      enc_prev = enc;
      have_prev = true;

      // PD control in encoder space → firmware force per axis
      std::array<int,3> f_enc = {0,0,0};
      for (int i = 0; i < 3; ++i) {
        double err = static_cast<double>(init_target_enc_[i] - enc[i]);
        double u = init_kp_ * err - init_kd_ * vel[i];
        // Clamp
        if (u > init_force_limit_) u = init_force_limit_;
        if (u < -init_force_limit_) u = -init_force_limit_;
        f_enc[i] = static_cast<int>(-u);
      }

      firmware_->setForces(f_enc);
      last_cmd_ = f_enc;

      if ((loops % 200) == 0) {
        RCLCPP_INFO(get_logger(), "Init loop %u | Enc: %6d %6d %6d | Target: %6d %6d %6d | Force: %5d %5d %5d",
                    loops, enc[0], enc[1], enc[2],
                    init_target_enc_[0], init_target_enc_[1], init_target_enc_[2],
                    f_enc[0], f_enc[1], f_enc[2]);
      }

      bool ok = (std::abs(init_target_enc_[0] - enc[0]) <= init_stable_eps_) &&
                 (std::abs(init_target_enc_[1] - enc[1]) <= init_stable_eps_) &&
                 (std::abs(init_target_enc_[2] - enc[2]) <= init_stable_eps_);
      if (ok) ++stable_count; else stable_count = 0;
      if (init_stable_count_req_ > 0 && static_cast<int>(stable_count) >= init_stable_count_req_) {
        RCLCPP_INFO(get_logger(), "Reached initial encoders (stable)");
        break;
      }
      ++loops;
    }

    // Stop forces after init routine
    firmware_->setForces({0,0,0});
    device_.runIOLoop();
  }

  // Params
  double force_scale_;
  double publish_rate_hz_;
  std::string frame_id_;
  int falcon_id_;
  int force_sensor_index_;
  // Init posture params
  bool init_posture_enable_ {true};
  std::array<int,3> init_target_enc_ {-500,-500,-500};
  double init_kp_ {100.0};
  double init_kd_ {0.1};
  int init_force_limit_ {1000};
  int init_max_loops_ {20000};
  int init_stable_eps_ {5};
  int init_stable_count_req_ {0};

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_force_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_force_array_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr pub_enc_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_pos_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Falcon device
  FalconDevice device_;
  std::shared_ptr<FalconFirmware> firmware_;
  bool falcon_initialized_;

  // State
  std::array<int,3> last_cmd_ {0,0,0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FalconNode>());
  rclcpp::shutdown();
  return 0;
}
