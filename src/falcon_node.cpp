#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <csignal>
#include <iostream>
#include <array>
#include <thread>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <ctime>
// ROS2 includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
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
 * Force 입력 토픽(/force_sensor/wrench, /force_sensor/wrench_array)을 받아
 * Novint Falcon 장치에 힘을 적용한다. (모든 publish 기능 제거됨)
 */
class FalconNode : public rclcpp::Node {
public:
  FalconNode() : Node("falcon_node"), falcon_initialized_(false) {
    // Parameters
  force_scale_ = this->declare_parameter<double>("force_scale", 0.0); // maps N to Falcon units (int16)
    if (force_scale_ == 0.0) {
      RCLCPP_WARN(get_logger(), "force_scale=0.0 => Falcon에 힘이 항상 0으로 적용됩니다. 파라미터 조정 필요.");
    }
  io_rate_hz_ = this->declare_parameter<double>("io_rate_hz", 1000.0); // 장치 IO 폴링 주파수 (Hz)
  // frame_id 파라미터 및 publish 관련 요소 제거됨
    falcon_id_ = this->declare_parameter<int>("falcon_id", 0); // which falcon to use (0-based)

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

    // Also support the combined array topic: layout [sensor, axis] (axis = fx,fy,fz,tx,ty,tz)
    sub_force_array_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/force_sensor/wrench_array", 10,
      std::bind(&FalconNode::on_force_array, this, std::placeholders::_1));

    // Force 적용 주기 (구독 수신 즉시 적용 대신 주기적으로 적용 가능)
    force_process_rate_hz_ = this->declare_parameter<double>("force_process_rate_hz", 200.0);
  // Safe mode & CSV params
  safe_mode_ = this->declare_parameter<bool>("safe_mode", true);
  csv_enable_ = this->declare_parameter<bool>("csv_enable", true);
  csv_dir_ = this->declare_parameter<std::string>("csv_dir", "");

    // Device init
    init_device();

    // (publish 제거) force 콜백 외에도 장치 주기 IO가 필요하므로 고주기 폴링 타이머 추가
    if (falcon_initialized_) {
      if (io_rate_hz_ < 1.0) io_rate_hz_ = 1.0;
      auto period = std::chrono::duration<double>(1.0 / io_rate_hz_);
      io_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&FalconNode::on_io_timer, this));
      // Force 처리 타이머
      if (force_process_rate_hz_ < 1.0) force_process_rate_hz_ = 1.0;
      auto fperiod = std::chrono::duration<double>(1.0 / force_process_rate_hz_);
      force_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(fperiod),
        std::bind(&FalconNode::on_force_timer, this));
      RCLCPP_INFO(get_logger(), "Falcon IO polling timer started at %.1f Hz", io_rate_hz_);
      RCLCPP_INFO(get_logger(), "Falcon force process timer started at %.1f Hz", force_process_rate_hz_);
    }
  }

  ~FalconNode() override {
    if (csv_file_.is_open()) {
      csv_file_.flush();
      csv_file_.close();
    }
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
    const int max_homing_attempts = 10000; // Increase timeout
    
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

  void on_force_array(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // Copy data so we can transform it
    auto data = msg->data;
    // Make all entries absolute values
    for (auto &v : data) {
      v = std::abs(v);
    }
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
    int idx = 0;
    if (idx >= sensors) idx = sensors - 1;
    const size_t off = static_cast<size_t>(idx * row_len);

    if (!is_ready()) {
      return; // only logging when device not ready
    }
    double sensor1_sum = data[off + 0] + data[off + 1] + data[off + 2] +
                          data[off + 3] + data[off + 4] + data[off + 5];
    size_t off2 = static_cast<size_t>((idx + 1) * row_len);
    size_t off3 = static_cast<size_t>((idx + 2) * row_len);
    double sensor2_sum = data[off2 + 0] + data[off2 + 1] + data[off2 + 2] + data[off2 + 3] + data[off2 + 4] + data[off2 + 5];
    double sensor3_sum = data[off3 + 0] + data[off3 + 1] + data[off3 + 2] + data[off3 + 3] + data[off3 + 4] + data[off3 + 5];
           
  // compact log
    static int counter = 0;
    if (++counter % 50 == 0) {
      RCLCPP_INFO(get_logger(), "Sensor sums | s1: %.3f, s2: %.3f, s3: %.3f", sensor1_sum, sensor2_sum, sensor3_sum);
    }

  // 최신 값 저장 (타이머에서 적용)
  last_sensor_sums_[0] = sensor1_sum;
  last_sensor_sums_[1] = sensor2_sum;
  last_sensor_sums_[2] = sensor3_sum;
  have_force_ = true;
  }

  // on_timer 제거됨 (publish 기능 삭제). 필요시 별도 주기 작업 구현 가능.
  void on_io_timer() {
    if (!is_ready()) return;
    if (!device_.runIOLoop()) return;
    // LED 간단 주기 토글 (너무 잦은 변경 방지 위해 카운터)
    static int cyc = 0;
    if (++cyc % static_cast<int>(io_rate_hz_) == 0) { // 1초마다
      int state = (cyc / static_cast<int>(io_rate_hz_)) % 2;
      firmware_->setLEDStatus(state ? 1 : 2);
    }
  }

  void on_force_timer() {
    if (!is_ready()) return;
    if (!have_force_) return; // 아직 데이터 없음
    apply_force_xyz(last_sensor_sums_[0], last_sensor_sums_[1], last_sensor_sums_[2]);
    // CSV logging with shared ROS time tick
    if (csv_enable_) {
      if (!csv_initialized_) init_csv();
      if (csv_file_.is_open()) {
        double t_sec = this->now().seconds();
        csv_file_ << std::fixed << std::setprecision(6)
                  << t_sec << ',' << force_counter_ << ','
                  << last_sensor_sums_[0] << ',' << last_sensor_sums_[1] << ',' << last_sensor_sums_[2] << ','
                  << last_cmd_[0] << ',' << last_cmd_[1] << ',' << last_cmd_[2] << '\n';
        if ((force_counter_ % 200) == 0) csv_file_.flush();
      }
    }
    ++force_counter_;
  }

  // Helper: scale, clamp, and send forces to device; update last_cmd_
  void apply_force_xyz(double sensor1_sum, double sensor2_sum, double sensor3_sum) {
    int v0 = static_cast<int>(std::round(sensor1_sum * force_scale_));
    int v1 = static_cast<int>(std::round(sensor2_sum * force_scale_));
    int v2 = static_cast<int>(std::round(sensor3_sum * force_scale_));
    int limit = 1000;
    v0 = std::max(-limit, std::min(limit, v0));
    v1 = std::max(-limit, std::min(limit, v1));
    v2 = std::max(-limit, std::min(limit, v2));
    if (safe_mode_) {
      v0 = v1 = v2 = 0;
    }
  firmware_->setForces({v0, v1, v2});
  // publish 제거로 인해 주기 IOLoop 호출이 사라졌으므로 즉시 IO 수행
  device_.runIOLoop();
    // RCLCPP_INFO(get_logger(), "Force command: [%d, %d, %d]", v0, v1, v2);
    last_cmd_ = {v0, v1, v2};   
    static int force_log_counter = 0;
    if (++force_log_counter % 1 == 0) {
      RCLCPP_INFO(get_logger(), "Force command: [%d, %d, %d]", v0, v1, v2);
      RCLCPP_INFO(get_logger(), "Sensor command: [%lf, %lf, %lf]", sensor1_sum, sensor2_sum, sensor3_sum);
    }
    // firmware_->setForces(f_enc);
  }

  void init_csv() {
    if (!csv_enable_ || csv_initialized_) return;
    std::string base_dir;
    if (!csv_dir_.empty()) {
      base_dir = csv_dir_;
    } else {
      auto t = std::time(nullptr);
      std::tm tm_buf{};
      localtime_r(&t, &tm_buf);
      char date_buf[16];
      std::strftime(date_buf, sizeof(date_buf), "%Y%m%d", &tm_buf);
      base_dir = std::string("outputs/falcon/") + date_buf;
    }
    std::string mkdir_cmd = std::string("mkdir -p ") + base_dir;
    int ret = std::system(mkdir_cmd.c_str()); (void)ret;
    auto t = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d_%H%M%S", &tm_buf);
    std::string path = base_dir + "/falcon_" + ts_buf + ".csv";
    csv_file_.open(path, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
      RCLCPP_ERROR(get_logger(), "CSV open 실패: %s", path.c_str());
      csv_enable_ = false;
      return;
    }
    csv_file_ << "t_sec,i,s1_sum,s2_sum,s3_sum,fx_cmd,fy_cmd,fz_cmd\n";
    csv_initialized_ = true;
    RCLCPP_INFO(get_logger(), "Falcon CSV logging -> %s", path.c_str());
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
        double err = static_cast<double>(init_target_enc_[i] - enc[i]); // 1: songwoo : target
        double u = init_kp_ * err - init_kd_ * vel[i];
        // Clamp
        if (u > init_force_limit_) u = init_force_limit_;
        if (u < -init_force_limit_) u = -init_force_limit_;
        f_enc[i] = static_cast<int>(-u);
      }
      bool safe_mode = true;
      if (safe_mode) {
        firmware_->setForces({0,0,0});
      } else {
        firmware_->setForces(f_enc);
      }
      last_cmd_ = f_enc;

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
  double io_rate_hz_ {1000.0};
  double force_process_rate_hz_ {200.0};
  bool safe_mode_ {true};
  bool csv_enable_ {true};
  std::string csv_dir_;
  int falcon_id_;
  // Init posture params
  bool init_posture_enable_ {true};
  std::array<int,3> init_target_enc_ {-500,-500,-500};
  double init_kp_ {100.0};
  double init_kd_ {0.1};
  int init_force_limit_ {1000};
  int init_max_loops_ {5000};
  int init_stable_eps_ {5};
  int init_stable_count_req_ {0};

  // ROS interfaces
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_force_array_;
  // 퍼블리셔 제거됨
  // 주기 IO 타이머
  rclcpp::TimerBase::SharedPtr io_timer_;
  rclcpp::TimerBase::SharedPtr force_timer_;

  // Falcon device
  FalconDevice device_;
  std::shared_ptr<FalconFirmware> firmware_;
  bool falcon_initialized_;
  std::array<double,3> last_sensor_sums_ {0.0,0.0,0.0};
  bool have_force_ {false};
  std::ofstream csv_file_;
  bool csv_initialized_ {false};
  size_t force_counter_ {0};

  // State
  std::array<int,3> last_cmd_ {0,0,0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FalconNode>());
  rclcpp::shutdown();
  return 0;
}
