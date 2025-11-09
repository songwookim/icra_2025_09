#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"

#ifdef HAVE_SENSEGLOVE_SDK
#include <SenseGlove/Core/HapticGlove.hpp>
#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HandPose.hpp>
#include <SenseGlove/Core/Vect3D.hpp>
#include <SenseGlove/Core/SenseCom.hpp>
#endif  // HAVE_SENSEGLOVE_SDK
#include <cstdlib>
namespace
{
constexpr std::size_t kFingerCount = 3;  // THUMB, INDEX, MIDDLE
constexpr std::size_t kJointsPerFinger = 3;
constexpr std::size_t kAxesPerSensor = 6;
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
    force_feedback_enabled_(false),
    force_feedback_sensor_index_(0),
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
    force_feedback_moments_{},
    filtered_ft_{},
    filtered_levels_{},
    filtered_strap_(0.0),
    filtered_vibe_(0.0),
    force_feedback_initialized_(false),
    force_feedback_levels_initialized_(false),
    latest_ft_valid_(false),
    last_ft_update_{0, 0, RCL_ROS_TIME},
    last_haptics_send_{0, 0, RCL_ROS_TIME}
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
        system("clear"); // Linux / macOS
        RCLCPP_INFO(get_logger(), "SenseGlove angles(rad): %s", oss.str().c_str());
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
  int force_feedback_sensor_index_;
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
  std::array<std::array<double, 3>, 4> force_feedback_moments_;
  std::array<double, kAxesPerSensor> filtered_ft_;
  std::array<double, kFingerCount> filtered_levels_;
  double filtered_strap_;
  double filtered_vibe_;
  bool force_feedback_initialized_;
  bool force_feedback_levels_initialized_;
  bool latest_ft_valid_;
  rclcpp::Time last_ft_update_;
  rclcpp::Time last_haptics_send_;
  rclcpp::Time last_pose_log_time_{0, 0, RCL_ROS_TIME};

  FingerJointMatrix joint_orientation_{};
  FingerJointMatrix zero_qpos_ref_{};
  FingerJointMatrix prev_qpos_cmd_{};
  FingerJointMatrix latest_smoothed_qpos_{};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr zero_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SenseGloveNode>());
  rclcpp::shutdown();
  return 0;
}
