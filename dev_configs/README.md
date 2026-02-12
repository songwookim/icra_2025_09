# Dev Configs

colcon 워크스페이스에서 사용하는 VS Code 설정 파일 백업입니다.

## 사용 방법

colcon 워크스페이스 루트(`~/ros2_ws/icra2025/`)에서 심볼릭 링크로 연결:

```bash
# 워크스페이스 루트로 이동
cd ~/ros2_ws/icra2025

# .vscode 폴더가 없으면 생성
mkdir -p .vscode

# 심볼릭 링크 (기존 파일이 있으면 덮어쓰기)
ln -sf $(pwd)/src/hri_falcon_robot_bridge/dev_configs/vscode/launch.json .vscode/launch.json
ln -sf $(pwd)/src/hri_falcon_robot_bridge/dev_configs/vscode/tasks.json .vscode/tasks.json
ln -sf $(pwd)/src/hri_falcon_robot_bridge/dev_configs/vscode/settings.json .vscode/settings.json
ln -sf $(pwd)/src/hri_falcon_robot_bridge/dev_configs/vscode/c_cpp_properties.json .vscode/c_cpp_properties.json
```

> `${workspaceFolder}`는 colcon 워크스페이스 루트를 가리킵니다.

## 포함 파일

| 파일 | 설명 |
|------|------|
| `launch.json` | 디버그 구성 (Python/C++ 노드별) |
| `tasks.json` | colcon build 태스크 |
| `settings.json` | 에디터 및 확장 설정 |
| `c_cpp_properties.json` | C++ IntelliSense 설정 |
