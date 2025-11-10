cmake_minimum_required(VERSION 3.8)
project(hri_falcon_robot_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Set libnifalcon paths (local build)
set(LIBNIFALCON_ROOT_DIR "/home/songwoo/Desktop/work_dir/libnifalcon")
set(LIBNIFALCON_INCLUDE_DIRS "${LIBNIFALCON_ROOT_DIR}/include")
set(LIBNIFALCON_LIBRARY_DIRS "${LIBNIFALCON_ROOT_DIR}/build/lib")
set(LIBNIFALCON_LIBRARIES "nifalcon")

# Check if libnifalcon exists
if(EXISTS "${LIBNIFALCON_INCLUDE_DIRS}/falcon/core/FalconDevice.h")
  set(LIBNIFALCON_FOUND TRUE)
  message(STATUS "Local libnifalcon found at ${LIBNIFALCON_ROOT_DIR}")
else()
  set(LIBNIFALCON_FOUND FALSE)
  message(STATUS "Local libnifalcon not found - building with simulation mode")
endif()

# --- C++ Falcon node ---
add_executable(falcon_node src/falcon_node.cpp)
ament_target_dependencies(falcon_node rclcpp std_msgs geometry_msgs)

# Always build falcon_node with debug symbols and no optimization for reliable debugging
if(NOT MSVC)
  target_compile_options(falcon_node PRIVATE -g -O0 -fno-omit-frame-pointer)
endif()

if(LIBNIFALCON_FOUND)
  message(STATUS "libnifalcon found - building with real Falcon support")
  target_include_directories(falcon_node PRIVATE ${LIBNIFALCON_INCLUDE_DIRS})
  target_link_directories(falcon_node PRIVATE ${LIBNIFALCON_LIBRARY_DIRS})
  target_link_libraries(falcon_node ${LIBNIFALCON_LIBRARIES})
  target_compile_definitions(falcon_node PRIVATE HAVE_LIBNIFALCON)
  # Ensure the runtime loader can find libnifalcon without manually exporting LD_LIBRARY_PATH
  set_target_properties(falcon_node PROPERTIES
    BUILD_RPATH "${LIBNIFALCON_LIBRARY_DIRS}"
    INSTALL_RPATH "${LIBNIFALCON_LIBRARY_DIRS}"
  )
else()
  message(STATUS "libnifalcon not found - building with simulation mode")
endif()

install(TARGETS falcon_node DESTINATION lib/${PROJECT_NAME})

# --- SenseGlove SDK (optional C++ node) ---
set(SENSEGLOVE_SDK_ROOT $ENV{SENSEGLOVE_SDK_ROOT})
if(NOT SENSEGLOVE_SDK_ROOT OR SENSEGLOVE_SDK_ROOT STREQUAL "")
  set(SENSEGLOVE_SDK_ROOT "/home/songwoo/Desktop/work_dir/SenseGlove-API")
endif()
set(SENSEGLOVE_INCLUDE_DIR "${SENSEGLOVE_SDK_ROOT}/include")
set(SENSEGLOVE_LIB_DIR $ENV{SENSEGLOVE_SDK_LIB_DIR})
if(NOT SENSEGLOVE_LIB_DIR OR SENSEGLOVE_LIB_DIR STREQUAL "")
  set(SENSEGLOVE_LIB_DIR "${SENSEGLOVE_SDK_ROOT}/lib/linux/v22/x86-64/release")
endif()

if(EXISTS "${SENSEGLOVE_INCLUDE_DIR}/SenseGlove/Core/HandLayer.hpp")
  set(SENSEGLOVE_SDK_FOUND TRUE)
  message(STATUS "SenseGlove SDK include found at ${SENSEGLOVE_INCLUDE_DIR}")
else()
  set(SENSEGLOVE_SDK_FOUND FALSE)
  message(WARNING "SenseGlove SDK include directory not found (expected ${SENSEGLOVE_INCLUDE_DIR}). sense_glove_node target will be skipped.")
endif()

if(SENSEGLOVE_SDK_FOUND AND EXISTS "${SENSEGLOVE_LIB_DIR}/libsgcore.so" AND EXISTS "${SENSEGLOVE_LIB_DIR}/libsgconnect.so")
  add_executable(sense_glove_node src/sense_glove_node.cpp)
  target_compile_definitions(sense_glove_node PRIVATE HAVE_SENSEGLOVE_SDK)
  target_include_directories(sense_glove_node PRIVATE ${SENSEGLOVE_INCLUDE_DIR})
  target_link_directories(sense_glove_node PRIVATE ${SENSEGLOVE_LIB_DIR})
  target_link_libraries(sense_glove_node sgcore sgconnect)
  ament_target_dependencies(sense_glove_node rclcpp sensor_msgs std_msgs std_srvs)
  if(NOT MSVC)
    target_compile_options(sense_glove_node PRIVATE -g -O0 -fno-omit-frame-pointer)
  endif()
  set_target_properties(sense_glove_node PROPERTIES
    BUILD_RPATH "${SENSEGLOVE_LIB_DIR}"
    INSTALL_RPATH "${SENSEGLOVE_LIB_DIR}"
  )
  install(TARGETS sense_glove_node DESTINATION lib/${PROJECT_NAME})
else()
  if(SENSEGLOVE_SDK_FOUND)
    message(WARNING "SenseGlove shared libraries not found under ${SENSEGLOVE_LIB_DIR}. sense_glove_node target will be skipped.")
  endif()
endif()

# --- Python nodes ---

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  hri_falcon_robot_bridge/force_sensor_node.py
  hri_falcon_robot_bridge/haptic_robot_controller.py
  hri_falcon_robot_bridge/hand_tracker_node.py
  hri_falcon_robot_bridge/sense_glove_mj_node.py
  hri_falcon_robot_bridge/robot_controller_node.py
  hri_falcon_robot_bridge/deformity_tracker_node.py
  hri_falcon_robot_bridge/emg_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Optional: install a .py-less alias so users can run `ros2 run ... hand_tracker_node`
install(PROGRAMS
  hri_falcon_robot_bridge/hand_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME hand_tracker_node
)

# Optional: alias for sense_glove_mj_node without .py
install(PROGRAMS
  hri_falcon_robot_bridge/sense_glove_mj_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sense_glove_mj_node
)

# Optional: alias for robot_controller_node without .py
install(PROGRAMS
  hri_falcon_robot_bridge/robot_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME robot_controller_node
)

# Optional: alias for deformity_tracker_node without .py
install(PROGRAMS
  hri_falcon_robot_bridge/deformity_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME deformity_tracker_node
)

# Optional: alias for emg_node without .py
install(PROGRAMS
  hri_falcon_robot_bridge/emg_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME emg_node
)

# --- Launch (optional) ---
# launch 디렉터리가 삭제되었을 경우 설치 단계에서 에러가 발생하지 않도록 조건부 처리
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
else()
  message(STATUS "[${PROJECT_NAME}] launch 디렉터리 없음: 설치 단계에서 생략")
endif()

# --- Resource files ---
install(DIRECTORY resource/ DESTINATION share/${PROJECT_NAME}/)

ament_package()
