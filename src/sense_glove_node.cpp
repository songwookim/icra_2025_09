#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "std_srvs/srv/trigger.hpp"

#ifdef HAVE_SENSEGLOVE_SDK
#include <SenseGlove/Core/HapticGlove.hpp>
#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HandPose.hpp>
#include <SenseGlove/Core/Vect3D.hpp>
#include <SenseGlove/Core/SenseCom.hpp>
#include <SenseGlove/Core/Nova2Glove.hpp>

#endif  // HAVE_SENSEGLOVE_SDK
#include <cstdlib>
namespace
{
constexpr std::size_t kFingerCount = 3;  // THUMB, INDEX, MIDDLE
constexpr std::size_t kJointsPerFinger = 3;
constexpr std::size_t kAxesPerSensor = 6;

struct ForceCalibrationSample
{
  float force_newton;
  float haptic_level;
};

struct JointLimit
{
  double open_rad;
  double closed_rad;
};

constexpr std::array<const char *, kFingerCount> kFingerNames{"THUMB", "INDEX", "MIDDLE"};
constexpr std::array<std::array<const char *, kJointsPerFinger>, kFingerCount> kJointNames{
  std::array<const char *, kJointsPerFinger>{"CMC", "MCP", "IP"},
  std::array<const char *, kJointsPerFinger>{"MCP", "PIP", "DIP"},
  std::array<const char *, kJointsPerFinger>{"MCP", "PIP", "DIP"}};

constexpr std::array<std::array<JointLimit, kJointsPerFinger>, kFingerCount> kJointLimits{{
  std::array<JointLimit, kJointsPerFinger>{
    JointLimit{0.0, 6.2831853071795862}, // 0~360도
    JointLimit{2.9670597283903604, 4.1887902047863905}, // approx 170~240도
    JointLimit{2.9670597283903604, 4.3633231299858238}},
  std::array<JointLimit, kJointsPerFinger>{
    JointLimit{3.0543261909901558, 4.1887902047863905},
    JointLimit{3.0543261909901558, 4.3633231299858238},
    JointLimit{3.0543261909901558, 4.4505895925855405}},
  std::array<JointLimit, kJointsPerFinger>{
    JointLimit{3.0543261909901558, 4.1887902047863905},
    JointLimit{3.0543261909901558, 4.4505895925855405},
    JointLimit{3.0543261909901558, 4.5378560551852573}}
}};

constexpr std::array<std::pair<const char *, const char *>, 9> kCommandOrder{{
  {"THUMB", "CMC"},
  {"THUMB", "MCP"},
  {"THUMB", "IP"},
  {"INDEX", "MCP"},
  {"INDEX", "PIP"},
  {"INDEX", "DIP"},
  {"MIDDLE", "MCP"},
  {"MIDDLE", "PIP"},
  {"MIDDLE", "DIP"},
}};

constexpr std::array<ForceCalibrationSample, 6> kCalibrationTable{{
  {0.0f, 0.0f},
  {3.0f, 0.25f},
  {5.0f, 0.4f},
  {8.0f, 0.6f},
  {12.0f, 0.85f},
  {18.0f, 1.0f},
}};

inline double clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

inline std::array<double, 3> cross_product(const std::array<double, 3> & a, const std::array<double, 3> & b)
{
  return std::array<double, 3>{
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
}

inline double magnitude3(const std::array<double, 3> & v)
{
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline int finger_index(const std::string & finger)
{
  for (std::size_t idx = 0; idx < kFingerNames.size(); ++idx)
  {
    if (finger == kFingerNames[idx])
    {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

inline int joint_index(std::size_t finger_idx, const std::string & joint)
{
  if (finger_idx >= kJointNames.size())
  {
    return -1;
  }
  for (std::size_t idx = 0; idx < kJointNames[finger_idx].size(); ++idx)
  {
    if (joint == kJointNames[finger_idx][idx])
    {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

inline float map_force_to_haptic_level(float force_newton)
{
  const float abs_force = std::abs(force_newton);
  
  if (abs_force <= kCalibrationTable.front().force_newton)
  {
    return kCalibrationTable.front().haptic_level;
  }
  
  if (abs_force >= kCalibrationTable.back().force_newton)
  {
    return kCalibrationTable.back().haptic_level;
  }
  
  for (std::size_t i = 1; i < kCalibrationTable.size(); ++i)
  {
    const auto& lower = kCalibrationTable[i - 1];
    const auto& upper = kCalibrationTable[i];
    
    if (abs_force <= upper.force_newton)
    {
      const float span = upper.force_newton - lower.force_newton;
      const float t = (abs_force - lower.force_newton) / span;
      return lower.haptic_level + t * (upper.haptic_level - lower.haptic_level);
    }
  }
  
  return kCalibrationTable.back().haptic_level;
}

}  // namespace

class SenseGloveNode : public rclcpp::Node
{
public:
  SenseGloveNode()
  : Node("sense_glove_node"),
    publish_joint_state_(true),
    qpos_gain_(0.5),
    qpos_smooth_alpha_(0.5),
    qpos_step_max_(0.05),
    clamp_qpos_symm_(true),
    clamp_qpos_min_(-1.57),
    clamp_qpos_max_(1.57),
    use_right_hand_(true),
    global_qpos_sign_(-1.0),
    glove_connected_logged_(false),
    latest_qpos_valid_(false),
    first_pose_logged_(false),
    log_interval_sec_(0.0),
    force_feedback_enabled_(true),
    force_feedback_sensor_count_(3),
    force_feedback_alpha_(0.15),
    force_feedback_level_alpha_(0.35),
    force_feedback_lambda_(0.3),
    force_feedback_force_norm_(15.0),
    force_feedback_torque_norm_(0.25),
    force_feedback_publish_rate_(60.0),
    force_feedback_publish_period_(1.0 / 60.0),
    force_feedback_fmax_{15.0, 18.0, 18.0, 12.0},
    force_feedback_dirs_{
      std::array<double, 3>{-1.0, 0.0, 0.0},
      std::array<double, 3>{-1.0, 0.2, 0.0},
      std::array<double, 3>{-1.0, 0.0, 0.2},
      std::array<double, 3>{-1.0, -0.2, 0.0}
    },
    force_feedback_positions_{
      std::array<double, 3>{0.02, -0.03, 0.0},
      std::array<double, 3>{0.03, 0.02, 0.0},
      std::array<double, 3>{0.04, 0.0, 0.03},
      std::array<double, 3>{0.03, -0.02, 0.02}
    },
    filtered_levels_{},
    filtered_strap_(0.0),
    filtered_vibe_(0.0),
    latest_ft_valid_(false),
    last_ft_update_{0, 0, RCL_ROS_TIME},
    last_haptics_send_{0, 0, RCL_ROS_TIME},
    all_sensors_ready_logged_(false)
  {
    declare_parameter("update_rate_hz", 50.0);
    declare_parameter("hand_side", std::string("right"));
    declare_parameter("publish_joint_state", true);
    declare_parameter("joint_state_topic", std::string("/hand_tracker/joint_state"));
    declare_parameter("qpos_gain", qpos_gain_);
    declare_parameter("qpos_smooth_alpha", qpos_smooth_alpha_);
    declare_parameter("qpos_step_max", qpos_step_max_);
    declare_parameter("clamp_qpos_symm", clamp_qpos_symm_);
    declare_parameter("clamp_qpos_min", clamp_qpos_min_);
    declare_parameter("clamp_qpos_max", clamp_qpos_max_);
    declare_parameter<double>("pose_log_interval_sec", 1.0);
  declare_parameter("force_feedback_topic_template", std::string("/force_sensor/s{index}/wrench"));
  declare_parameter("force_feedback_sensor_count", force_feedback_sensor_count_);
    declare_parameter("force_feedback_alpha", force_feedback_alpha_);
    declare_parameter("force_feedback_level_alpha", force_feedback_level_alpha_);
    declare_parameter("force_feedback_lambda", force_feedback_lambda_);
    declare_parameter("force_feedback_force_norm", force_feedback_force_norm_);
    declare_parameter("force_feedback_torque_norm", force_feedback_torque_norm_);
    declare_parameter("force_feedback_publish_rate", force_feedback_publish_rate_);
    log_interval_sec_ = std::max(0.0, get_parameter("pose_log_interval_sec").as_double());

    const double raw_update_rate = get_parameter("update_rate_hz").as_double();
    update_rate_hz_ = raw_update_rate < 1.0 ? 1.0 : raw_update_rate;

    const auto hand_side_raw = get_parameter("hand_side").as_string();
    std::string hand_side_lower = hand_side_raw;
    std::transform(hand_side_lower.begin(), hand_side_lower.end(), hand_side_lower.begin(), ::tolower);
    use_right_hand_ = !(hand_side_lower == "left" || hand_side_lower == "l");

    publish_joint_state_ = get_parameter("publish_joint_state").as_bool();
    joint_state_topic_ = get_parameter("joint_state_topic").as_string();
    qpos_gain_ = get_parameter("qpos_gain").as_double();
    qpos_smooth_alpha_ = clamp(get_parameter("qpos_smooth_alpha").as_double(), 0.0, 1.0);
    qpos_step_max_ = std::max(0.0, get_parameter("qpos_step_max").as_double());
    clamp_qpos_symm_ = get_parameter("clamp_qpos_symm").as_bool();
    clamp_qpos_min_ = get_parameter("clamp_qpos_min").as_double();
    clamp_qpos_max_ = get_parameter("clamp_qpos_max").as_double();

    force_feedback_topic_template_ = get_parameter("force_feedback_topic_template").as_string();
    force_feedback_sensor_count_ = get_parameter("force_feedback_sensor_count").as_int();
    if (force_feedback_sensor_count_ < 1)
    {
      force_feedback_sensor_count_ = 1;
    }
    const int max_supported_sensors = static_cast<int>(kFingerCount);
    if (force_feedback_sensor_count_ > max_supported_sensors)
    {
      RCLCPP_WARN(get_logger(),
        "Force feedback supports up to %d sensors; clamping requested count %d",
        max_supported_sensors, force_feedback_sensor_count_);
      force_feedback_sensor_count_ = max_supported_sensors;
    }
    force_feedback_alpha_ = clamp(get_parameter("force_feedback_alpha").as_double(), 0.0, 1.0);
    force_feedback_level_alpha_ = clamp(get_parameter("force_feedback_level_alpha").as_double(), 0.0, 1.0);
    force_feedback_lambda_ = clamp(get_parameter("force_feedback_lambda").as_double(), 0.0, 1.0);
    force_feedback_force_norm_ = get_parameter("force_feedback_force_norm").as_double();
    force_feedback_torque_norm_ = get_parameter("force_feedback_torque_norm").as_double();
    const double publish_rate_param = get_parameter("force_feedback_publish_rate").as_double();
    if (publish_rate_param > 1e-3)
    {
      force_feedback_publish_rate_ = publish_rate_param;
    }
    force_feedback_publish_period_ = 1.0 / std::max(force_feedback_publish_rate_, 1e-3);
    force_feedback_force_norm_ = std::max(force_feedback_force_norm_, 1e-6);
    force_feedback_torque_norm_ = std::max(force_feedback_torque_norm_, 1e-6);

    const std::vector<double> default_calib_force{0.0, 3.0, 5.0, 8.0, 12.0, 18.0};
    const std::vector<double> default_calib_level{0.0, 0.25, 0.4, 0.6, 0.85, 1.0};
    const auto calib_force_param = declare_parameter<std::vector<double>>(
      "force_feedback_calibration_force", default_calib_force);
    const auto calib_level_param = declare_parameter<std::vector<double>>(
      "force_feedback_calibration_level", default_calib_level);

    force_feedback_calibration_.clear();
    if (calib_force_param.size() == calib_level_param.size() && calib_force_param.size() >= 2)
    {
      force_feedback_calibration_.reserve(calib_force_param.size());
      for (std::size_t idx = 0; idx < calib_force_param.size(); ++idx)
      {
        const double force_sample = std::max(0.0, calib_force_param[idx]);
        const double level_sample = clamp(calib_level_param[idx], 0.0, 1.0);
        force_feedback_calibration_.emplace_back(force_sample, level_sample);
      }
      std::sort(force_feedback_calibration_.begin(), force_feedback_calibration_.end(),
        [](const auto & a, const auto & b) { return a.first < b.first; });
    }
    else
    {
      RCLCPP_WARN(get_logger(),
        "Force feedback calibration table invalid; falling back to simple normalization.");
    }

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_state_topic_, 10);
    zero_srv_ = create_service<std_srvs::srv::Trigger>(
      "set_zero", std::bind(&SenseGloveNode::handle_zero_request, this, std::placeholders::_1,
      std::placeholders::_2));

    for (auto & row : joint_orientation_)
    {
      row.fill(-1.0);
    }
    for (auto & row : zero_qpos_ref_)
    {
      row.fill(0.0);
    }
    for (auto & row : prev_qpos_cmd_)
    {
      row.fill(0.0);
    }
    for (auto & row : latest_smoothed_qpos_)
    {
      row.fill(0.0);
    }
    filtered_levels_.fill(0.0);
    filtered_strap_ = 0.0;
    filtered_vibe_ = 0.0;
    const std::size_t sensor_count = static_cast<std::size_t>(force_feedback_sensor_count_);
    filtered_sensor_ft_.assign(sensor_count, std::array<double, kAxesPerSensor>{});
    sensor_initialized_.assign(sensor_count, false);
    last_sensor_update_.assign(sensor_count, rclcpp::Time(0, 0, RCL_ROS_TIME));
    force_feedback_topics_.clear();
    force_feedback_topics_.reserve(sensor_count);
    force_wrench_subs_.clear();
    force_wrench_subs_.reserve(sensor_count);
    all_sensors_ready_logged_ = false;

    if (force_feedback_enabled_)
    {
      auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
      for (std::size_t sensor_idx = 0; sensor_idx < sensor_count; ++sensor_idx)
      {
        const std::string topic = resolve_force_feedback_topic(static_cast<int>(sensor_idx));
        force_feedback_topics_.push_back(topic);
        auto cb = [this, sensor_idx](const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
        {
          on_force_wrench(sensor_idx, msg);
        };
        force_wrench_subs_.emplace_back(create_subscription<geometry_msgs::msg::WrenchStamped>(
          topic, qos, cb));
      }

      if (!force_feedback_topics_.empty())
      {
        std::ostringstream oss;
        for (std::size_t idx = 0; idx < force_feedback_topics_.size(); ++idx)
        {
          if (idx > 0)
          {
            oss << ", ";
          }
          oss << force_feedback_topics_[idx];
        }
        const std::string topic_list = oss.str();
        RCLCPP_INFO(
          get_logger(),
          "Force feedback enabled; listening to %zu sensor topics: %s",
          force_feedback_topics_.size(), topic_list.c_str());
      }

      force_feedback_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(force_feedback_publish_period_)),
        std::bind(&SenseGloveNode::send_force_feedback, this));
    }

#ifdef HAVE_SENSEGLOVE_SDK
    if (SGCore::SenseCom::ScanningActive())
    {
      RCLCPP_INFO(get_logger(), "SenseCom scanning already active.");
    }
    else if (SGCore::SenseCom::StartupSenseCom())
    {
      RCLCPP_INFO(get_logger(), "SenseCom scanning started; waiting for gloves...");
    }
    else
    {
      RCLCPP_WARN(get_logger(), "Failed to start SenseCom; gloves may not appear.");
    }
#else
    RCLCPP_WARN(get_logger(),
      "SenseGlove SDK not available at build time. Node will spin but no glove data can be read.");
#endif

    const auto period = std::chrono::duration<double>(1.0 / update_rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&SenseGloveNode::on_timer, this));
  }

private:
  using FingerJointMatrix = std::array<std::array<double, kJointsPerFinger>, kFingerCount>;

  void on_force_wrench(std::size_t sensor_index, const geometry_msgs::msg::WrenchStamped::SharedPtr msg);
  void send_force_feedback();
  std::string resolve_force_feedback_topic(int sensor_index) const;

  void handle_zero_request(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> & /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!latest_qpos_valid_)
    {
      response->success = false;
      response->message = "No SenseGlove data received yet.";
      return;
    }

    zero_qpos_ref_ = latest_smoothed_qpos_;
    for (auto & row : prev_qpos_cmd_)
    {
      row.fill(0.0);
    }
    response->success = true;
    response->message = "Zero reference captured from current glove pose.";
    RCLCPP_INFO(get_logger(), "Zero offsets updated from SenseGlove pose.");
  }

  void on_timer()
  {
    const auto angles_opt = read_joint_angles();
    if (!angles_opt.has_value())
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No SenseGlove pose data available yet.");
      return;
    }

    const auto angles = angles_opt.value();
    process_command_stream(angles);

    if (!publish_joint_state_)
    {
      return;
    }

    sensor_msgs::msg::JointState joint_msg;
    joint_msg.header.stamp = get_clock()->now();
    joint_msg.name.reserve(kFingerCount * kJointsPerFinger);
    joint_msg.position.reserve(kFingerCount * kJointsPerFinger);

    for (std::size_t f = 0; f < kFingerCount; ++f)
    {
      for (std::size_t j = 0; j < kJointsPerFinger; ++j)
      {
        const double qpos = latest_smoothed_qpos_[f][j];
        if (!std::isfinite(qpos))
        {
          continue;
        }
        joint_msg.name.emplace_back(std::string(kFingerNames[f]) + "_" + kJointNames[f][j]);
        joint_msg.position.emplace_back(qpos);
      }
    }

    joint_pub_->publish(joint_msg);
  }

  void process_command_stream(const FingerJointMatrix & angles)
  {
    FingerJointMatrix smoothed_qpos = latest_smoothed_qpos_;

    for (const auto & entry : kCommandOrder)
    {
      const int f_idx = finger_index(entry.first);
      const int j_idx = f_idx >= 0 ? joint_index(static_cast<std::size_t>(f_idx), entry.second) : -1;
      double raw_angle_rad = std::numeric_limits<double>::quiet_NaN();
      if (f_idx >= 0 && j_idx >= 0)
      {
        raw_angle_rad = angles[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)];
      }

      double qpos = map_angle_to_qpos(f_idx, j_idx, raw_angle_rad);
      if (f_idx >= 0 && j_idx >= 0)
      {
        qpos -= zero_qpos_ref_[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)];

        const double previous = prev_qpos_cmd_[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)];
        double smoothed = (1.0 - qpos_smooth_alpha_) * previous + qpos_smooth_alpha_ * qpos;
        const double delta = smoothed - previous;
        if (delta > qpos_step_max_)
        {
          smoothed = previous + qpos_step_max_;
        }
        else if (delta < -qpos_step_max_)
        {
          smoothed = previous - qpos_step_max_;
        }
        if (clamp_qpos_symm_)
        {
          smoothed = clamp(smoothed, clamp_qpos_min_, clamp_qpos_max_);
        }

        prev_qpos_cmd_[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)] = smoothed;
        smoothed_qpos[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)] = smoothed;
      }
    }

    latest_smoothed_qpos_ = smoothed_qpos;
    latest_qpos_valid_ = true;

    const auto now = get_clock()->now();
    const bool should_log_pose =
      (!first_pose_logged_) ||
      ((log_interval_sec_ > 0.0) && (last_pose_log_time_.nanoseconds() > 0) &&
       ((now - last_pose_log_time_).seconds() >= log_interval_sec_));

    if (should_log_pose)
    {
      std::ostringstream oss;
      bool any = false;
      for (std::size_t idx = 0; idx < kCommandOrder.size(); ++idx)
      {
        const auto & entry = kCommandOrder[idx];
        const int f_idx = finger_index(entry.first);
        const int j_idx = f_idx >= 0 ? joint_index(static_cast<std::size_t>(f_idx), entry.second) : -1;
        if (f_idx < 0 || j_idx < 0)
        {
          continue;
        }
        const double angle_rad = angles[static_cast<std::size_t>(f_idx)][static_cast<std::size_t>(j_idx)];
        if (std::isnan(angle_rad))
        {
          continue;
        }
        if (any)
        {
          oss << ", ";
        }
        oss << entry.first << '_' << entry.second << '=' << std::fixed << std::setprecision(3) << angle_rad;
        any = true;
      }
      if (any)
      {
        //RCLCPP_INFO(get_logger(), "SenseGlove angles(rad): %s", oss.str().c_str());
        NULL;
      }
      first_pose_logged_ = true;
      last_pose_log_time_ = now;
    }
  }

  std::optional<FingerJointMatrix> read_joint_angles()
  {
#ifdef HAVE_SENSEGLOVE_SDK
    SGCore::HandPose pose;
    if (!SGCore::HandLayer::GetHandPose(use_right_hand_, pose))
    {
      static bool connection_warned = false;
      const int connected = SGCore::HandLayer::GlovesConnected();
      const bool device_on_hand = SGCore::HandLayer::DeviceConnected(use_right_hand_);
      if (!connection_warned && (connected == 0 || !device_on_hand))
      {
        connection_warned = true;
        RCLCPP_WARN(get_logger(),
          "SenseGlove not detected (GlovesConnected=%d, DeviceConnected=%s). Ensure SenseCom is running and the glove is powered.",
          connected, device_on_hand ? "true" : "false");
      }
      return std::nullopt;
    }

    FingerJointMatrix result;
    for (auto & row : result)
    {
      row.fill(std::numeric_limits<double>::quiet_NaN());
    }

    bool any_valid = false;
    const auto & hand_angles = pose.GetHandAngles();
    const std::size_t finger_count = std::min<std::size_t>(hand_angles.size(), kFingerCount);

    for (std::size_t f = 0; f < finger_count; ++f)
    {
      const auto & joints = hand_angles[f];
      const std::size_t joint_count = std::min<std::size_t>(joints.size(), kJointsPerFinger);
      for (std::size_t j = 0; j < joint_count; ++j)
      {
        const double flex_rad = static_cast<double>(joints[j].GetY());  // Y-axis is flexion/extension per SDK docs.
        if (!std::isfinite(flex_rad))
        {
          continue;
        }
        result[f][j] = flex_rad;
        any_valid = true;
      }
    }

    if (!any_valid)
    {
      std::array<double, 4> normalized{};
      normalized.fill(-1.0);
      const auto flex = pose.GetNormalizedFlexion(true);
      const std::size_t copy_count = std::min<std::size_t>(normalized.size(), flex.size());
      for (std::size_t idx = 0; idx < copy_count; ++idx)
      {
        normalized[idx] = clamp(static_cast<double>(flex[idx]), 0.0, 1.0);
      }
      const auto fallback = flex_to_angles(normalized);
      bool fallback_valid = false;
      for (const auto & row : fallback)
      {
        for (double angle_rad : row)
        {
          if (std::isfinite(angle_rad))
          {
            fallback_valid = true;
            break;
          }
        }
        if (fallback_valid)
        {
          break;
        }
      }

      if (!fallback_valid)
      {
        return std::nullopt;
      }

      if (!glove_connected_logged_)
      {
        RCLCPP_INFO(get_logger(), "SenseGlove connected (%s hand detected).",
          use_right_hand_ ? "right" : "left");
        glove_connected_logged_ = true;
      }
      return fallback;
    }

    if (!glove_connected_logged_)
    {
      RCLCPP_INFO(get_logger(), "SenseGlove connected (%s hand detected).",
        use_right_hand_ ? "right" : "left");
      glove_connected_logged_ = true;
    }

    return result;
#else
    return std::nullopt;
#endif
  }

  FingerJointMatrix flex_to_angles(const std::array<double, 4> & flex) const
  {
    FingerJointMatrix result;
    for (auto & row : result)
    {
      row.fill(std::numeric_limits<double>::quiet_NaN());
    }

    for (std::size_t f = 0; f < kFingerCount; ++f)
    {
      const double flex_val = flex[f];
      if (flex_val < 0.0)
      {
        continue;
      }
      for (std::size_t j = 0; j < kJointsPerFinger; ++j)
      {
        const auto & limits = kJointLimits[f][j];
        const double angle_rad = limits.open_rad + flex_val * (limits.closed_rad - limits.open_rad);
        result[f][j] = angle_rad;
      }
    }

    return result;
  }

  double map_force_to_level(double force_newton) const
  {
    const double non_negative_force = std::max(0.0, force_newton);

    if (!force_feedback_calibration_.empty())
    {
      if (non_negative_force <= force_feedback_calibration_.front().first)
      {
        return force_feedback_calibration_.front().second;
      }
      if (non_negative_force >= force_feedback_calibration_.back().first)
      {
        return force_feedback_calibration_.back().second;
      }

      for (std::size_t idx = 1; idx < force_feedback_calibration_.size(); ++idx)
      {
        const auto & lower = force_feedback_calibration_[idx - 1];
        const auto & upper = force_feedback_calibration_[idx];
        if (non_negative_force <= upper.first)
        {
          const double span = upper.first - lower.first;
          const double t = span > 1e-6 ? (non_negative_force - lower.first) / span : 0.0;
          const double level = lower.second + t * (upper.second - lower.second);
          return clamp(level, 0.0, 1.0);
        }
      }
    }

    if (force_feedback_force_norm_ <= 0.0)
    {
      return 0.0;
    }

    const double normalized = non_negative_force / force_feedback_force_norm_;
    return clamp(normalized, 0.0, 1.0);
  }

  double map_angle_to_qpos(int finger_idx, int joint_idx, double raw_angle_rad) const
  {
    if (finger_idx < 0 || joint_idx < 0 || std::isnan(raw_angle_rad))
    {
      return 0.0;
    }

    double centered_rad = raw_angle_rad;
    if (raw_angle_rad > M_PI || raw_angle_rad < -M_PI)
    {
      centered_rad = std::fmod(raw_angle_rad + M_PI, 2.0 * M_PI);
      if (centered_rad < 0.0)
      {
        centered_rad += 2.0 * M_PI;
      }
      centered_rad -= M_PI;
    }
    const double direction = joint_orientation_[static_cast<std::size_t>(finger_idx)][static_cast<std::size_t>(joint_idx)];
    double qpos = centered_rad * direction * global_qpos_sign_;
    qpos *= qpos_gain_;
    if (clamp_qpos_symm_)
    {
      qpos = clamp(qpos, clamp_qpos_min_, clamp_qpos_max_);
    }
    return qpos;
  }

  // Parameters & state
  double update_rate_hz_;
  bool publish_joint_state_;
  std::string joint_state_topic_;
  double qpos_gain_;
  double qpos_smooth_alpha_;
  double qpos_step_max_;
  bool clamp_qpos_symm_;
  double clamp_qpos_min_;
  double clamp_qpos_max_;

  bool use_right_hand_;
  double global_qpos_sign_;
  bool glove_connected_logged_;
  bool latest_qpos_valid_;
  bool first_pose_logged_;
  double log_interval_sec_;
  bool force_feedback_enabled_;
  int force_feedback_sensor_count_;
  double force_feedback_alpha_;
  double force_feedback_level_alpha_;
  double force_feedback_lambda_;
  double force_feedback_force_norm_;
  double force_feedback_torque_norm_;
  double force_feedback_publish_rate_;
  double force_feedback_publish_period_;
  std::array<double, 4> force_feedback_fmax_;
  std::array<std::array<double, 3>, 4> force_feedback_dirs_;
  std::array<std::array<double, 3>, 4> force_feedback_positions_;
  std::array<double, kFingerCount> filtered_levels_;
  double filtered_strap_;
  double filtered_vibe_;
  bool latest_ft_valid_;
  rclcpp::Time last_ft_update_;
  rclcpp::Time last_haptics_send_;
  rclcpp::Time last_pose_log_time_{0, 0, RCL_ROS_TIME};
  std::array<double, kFingerCount> last_sent_levels_{};  // tracks last sent levels to avoid redundant sends

  std::vector<std::array<double, kAxesPerSensor>> filtered_sensor_ft_;
  std::vector<bool> sensor_initialized_;
  std::vector<rclcpp::Time> last_sensor_update_;
  std::vector<std::string> force_feedback_topics_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr> force_wrench_subs_;
  bool all_sensors_ready_logged_;
  std::vector<std::pair<double, double>> force_feedback_calibration_;

  FingerJointMatrix joint_orientation_{};
  FingerJointMatrix zero_qpos_ref_{};
  FingerJointMatrix prev_qpos_cmd_{};
  FingerJointMatrix latest_smoothed_qpos_{};

  std::string force_feedback_topic_template_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr zero_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr force_feedback_timer_;
  std::mutex force_feedback_mutex_;
};

std::string SenseGloveNode::resolve_force_feedback_topic(int sensor_index) const
{
  std::string resolved = force_feedback_topic_template_;
  const std::string placeholder = "{index}";
  const int one_based = sensor_index + 1;
  bool replaced = false;

  std::size_t pos = resolved.find(placeholder);
  while (pos != std::string::npos)
  {
    resolved.replace(pos, placeholder.size(), std::to_string(one_based));
    replaced = true;
    pos = resolved.find(placeholder, pos + 1);
  }

  if (!replaced)
  {
    if (resolved.empty())
    {
      resolved = "/force_sensor/s" + std::to_string(one_based) + "/wrench";
    }
    else if (resolved.back() == '/')
    {
      resolved += "s" + std::to_string(one_based) + "/wrench";
    }
    else
    {
      resolved += "/s" + std::to_string(one_based) + "/wrench";
    }
  }

  return resolved;
}

void SenseGloveNode::on_force_wrench(
  std::size_t sensor_index, const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
{
  if (!force_feedback_enabled_ || msg == nullptr)
  {
    return;
  }

  if (sensor_index >= filtered_sensor_ft_.size())
  {
    return;
  }

  const auto now = get_clock()->now();
  std::array<double, kAxesPerSensor> sample{
    msg->wrench.force.x,
    msg->wrench.force.y,
    msg->wrench.force.z,
    msg->wrench.torque.x,
    msg->wrench.torque.y,
    msg->wrench.torque.z
  };
  const std::size_t target_finger = std::min(sensor_index, static_cast<std::size_t>(kFingerCount - 1));

  {
    std::lock_guard<std::mutex> guard(force_feedback_mutex_);

    auto & filtered_sample = filtered_sensor_ft_[sensor_index];
    if (!sensor_initialized_[sensor_index])
    {
      filtered_sample = sample;
      sensor_initialized_[sensor_index] = true;
    }
    else
    {
      for (std::size_t axis = 0; axis < kAxesPerSensor; ++axis)
      {
        const double previous = filtered_sample[axis];
        filtered_sample[axis] = (1.0 - force_feedback_alpha_) * previous + force_feedback_alpha_ * sample[axis];
      }
    }

    last_sensor_update_[sensor_index] = now;

    const double force_mag = std::sqrt(
      std::abs(filtered_sample[0]) * std::abs(filtered_sample[0]) +
      std::abs(filtered_sample[1]) * std::abs(filtered_sample[1]) +
      std::abs(filtered_sample[2]) * std::abs(filtered_sample[2]));
    const double torque_mag = std::sqrt(
      std::abs(filtered_sample[3]) * std::abs(filtered_sample[3]) +
      std::abs(filtered_sample[4]) * std::abs(filtered_sample[4]) +
      std::abs(filtered_sample[5]) * std::abs(filtered_sample[5]));

    const double target_level = static_cast<double>(map_force_to_haptic_level(static_cast<float>(force_mag)));
    const double normalized_torque = clamp(torque_mag, 0.0, 1.0);

    const double previous_level = filtered_levels_[target_finger];
    filtered_levels_[target_finger] =
      (1.0 - force_feedback_level_alpha_) * previous_level + force_feedback_level_alpha_ * target_level;

    double aggregated_force = 0.0;
    double aggregated_torque = 0.0;
    int active_count = 0;
    for (std::size_t idx = 0; idx < filtered_sensor_ft_.size(); ++idx)
    {
      if (!sensor_initialized_[idx])
      {
        continue;
      }
      const auto & ft = filtered_sensor_ft_[idx];
      aggregated_force += std::sqrt(
        std::abs(ft[0]) * std::abs(ft[0]) + 
        std::abs(ft[1]) * std::abs(ft[1]) + 
        std::abs(ft[2]) * std::abs(ft[2]));
      aggregated_torque += std::sqrt(
        std::abs(ft[3]) * std::abs(ft[3]) + 
        std::abs(ft[4]) * std::abs(ft[4]) + 
        std::abs(ft[5]) * std::abs(ft[5]));
      ++active_count;
    }

    if (active_count > 0)
    {
      aggregated_force /= static_cast<double>(active_count);
      aggregated_torque /= static_cast<double>(active_count);
      const double aggregated_force_level = clamp(aggregated_force, 0.0, 1.0);
      const double normalized_aggregate_torque = clamp(aggregated_torque, 0.0, 1.0);
      filtered_strap_ = (1.0 - force_feedback_lambda_) * filtered_strap_ +
        force_feedback_lambda_ * aggregated_force_level;
      filtered_vibe_ = (1.0 - force_feedback_lambda_) * filtered_vibe_ +
        force_feedback_lambda_ * normalized_aggregate_torque;
      latest_ft_valid_ = true;
      last_ft_update_ = now;
    }
    else
    {
      latest_ft_valid_ = false;
    }

    if (!all_sensors_ready_logged_)
    {
      const bool all_ready = std::all_of(sensor_initialized_.begin(), sensor_initialized_.end(),
        [](bool ready) { return ready; });
      if (all_ready)
      {
        std::ostringstream oss;
        for (std::size_t idx = 0; idx < sensor_initialized_.size(); ++idx)
        {
          if (idx > 0)
          {
            oss << ", ";
          }
          const std::size_t finger_map = std::min(idx, static_cast<std::size_t>(kFingerCount - 1));
          oss << "s" << (idx + 1) << "->" << kFingerNames[finger_map];
        }
        const std::string mapping = oss.str();
        RCLCPP_INFO(
          get_logger(),
          "Force feedback sensors active (%zu): %s",
          sensor_initialized_.size(), mapping.c_str());
        all_sensors_ready_logged_ = true;
      }
    }
  }

  // RCLCPP_INFO(
  //   get_logger(),
  //   "Force sensor[%zu -> %s] wrench: f(%.3f, %.3f, %.3f) t(%.3f, %.3f, %.3f)",
  //   sensor_index + 1,
  //   kFingerNames[target_finger],
  //   sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]);
}

void SenseGloveNode::send_force_feedback()
{
  if (!force_feedback_enabled_)
  {
    return;
  }

  const auto now = get_clock()->now();

  std::array<double, kFingerCount> levels{};
  std::array<double, kFingerCount> forces_newton{};
  double strap_level = 0.0;
  double vibe_level = 0.0;

  {
    std::lock_guard<std::mutex> guard(force_feedback_mutex_);
    const double stale_threshold = 0.5;
    bool any_active = false;

    for (std::size_t idx = 0; idx < sensor_initialized_.size(); ++idx)
    {
      const double age = (now - last_sensor_update_[idx]).seconds();
      const std::size_t finger = std::min(idx, static_cast<std::size_t>(kFingerCount - 1));
      if (!sensor_initialized_[idx] || age > stale_threshold)
      {
        if (sensor_initialized_[idx] && age > stale_threshold)
        {
          sensor_initialized_[idx] = false;
          filtered_sensor_ft_[idx].fill(0.0);
        }
        filtered_levels_[finger] = (1.0 - force_feedback_level_alpha_) * filtered_levels_[finger];
        forces_newton[finger] = 0.0;
        continue;
      }
      any_active = true;
      
      const auto & ft = filtered_sensor_ft_[idx];
      forces_newton[finger] = std::sqrt(
        std::abs(ft[0]) * std::abs(ft[0]) + 
        std::abs(ft[1]) * std::abs(ft[1]) + 
        std::abs(ft[2]) * std::abs(ft[2]));
    }

    if (!any_active)
    {
      filtered_levels_.fill(0.0);
      filtered_strap_ = 0.0;
      filtered_vibe_ = 0.0;
      latest_ft_valid_ = false;
      forces_newton.fill(0.0);
    }

    levels = filtered_levels_;
    strap_level = filtered_strap_;
    vibe_level = filtered_vibe_;
  }

#ifndef HAVE_SENSEGLOVE_SDK
  (void)levels;
  (void)strap_level;
  (void)vibe_level;
  static bool warned = false;
  if (!warned)
  {
    RCLCPP_WARN(get_logger(),
      "SenseGlove SDK not available; force feedback disabled.");
    warned = true;
  }
  last_haptics_send_ = now;
  return;
#else
  try
  {
    bool queued_thumb = SGCore::HandLayer::QueueCommand_ForceFeedbackLevel(
      use_right_hand_, 0, static_cast<float>(clamp(levels[0], 0.0, 1.0)), false);
    
    bool queued_index = true;
    if (kFingerCount > 1)
    {
      queued_index = SGCore::HandLayer::QueueCommand_ForceFeedbackLevel(
        use_right_hand_, 1, static_cast<float>(clamp(levels[1], 0.0, 1.0)), false);
    }
    
    bool queued_middle = true;
    if (kFingerCount > 2)
    {
      queued_middle = SGCore::HandLayer::QueueCommand_ForceFeedbackLevel(
        use_right_hand_, 2, static_cast<float>(clamp(levels[2], 0.0, 1.0)), false);
    }

    const bool all_queued = queued_thumb && queued_index && queued_middle;

    if (!all_queued || !SGCore::HandLayer::SendHaptics(use_right_hand_))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to queue or send haptic commands.");
      return;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "Target fingertip forces (N): thumb=%.2f, index=%.2f, middle=%.2f",
      forces_newton[0],
      forces_newton[1],
      forces_newton[2]);

    last_haptics_send_ = now;
  }
  catch (const std::exception & exc)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Failed to forward force feedback to SenseGlove: %s", exc.what());
  }
#endif
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SenseGloveNode>());
  rclcpp::shutdown();
  return 0;
}
