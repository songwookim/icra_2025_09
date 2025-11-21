// 간단/지속 루프: 라이브러리 문자열 반환 함수(Version 등) 호출 제거하여 ABI 문제 최소화.
// libc++ 빌드된 SGCore와 호환 위해 clang++ -stdlib=libc++ 권장.

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#include <SenseGlove/Core/Nova2Glove.hpp>
#include <SenseGlove/Core/SenseCom.hpp>

int main() {
    if (!SGCore::SenseCom::ScanningActive()) {
        SGCore::SenseCom::StartupSenseCom();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    const bool rightHand = true; // 필요 시 false로 변경
    const std::array<float, 4> kTestLevels = {0.0f, 0.3f, 0.6f, 1.0f};
    // seconds(0.0002) 는 정수 duration이라 변환 불가 → microseconds 사용 (200us)
    const auto holdDuration = std::chrono::microseconds(2000000); // 0.0002s
    std::size_t levelIndex = 0;

    std::cout << "[Loop] Nova2 직접 제어 테스트 (Thumb FFB)" << std::endl;
    std::cout << "      2초마다 level=0,0.3,0.6,1.0 순으로 반복" << std::endl;

    while (true) {
        SGCore::Nova::Nova2Glove glove;
        if (!SGCore::Nova::Nova2Glove::GetNova2Glove(rightHand, glove)) {
            std::cout << "장치 연결 대기중..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        const float level = kTestLevels[levelIndex];
        glove.QueueForceFeedbackLevel(0, level); // Thumb
        // 필요 시 비교를 위해 다른 손가락 활성화. 지금은 의도 명확화를 위해 Thumb만.
        glove.SendHaptics();

        std::cout << "[Thumb] level -> " << level << std::endl;
        levelIndex = (levelIndex + 1) % kTestLevels.size();

        std::this_thread::sleep_for(holdDuration);
    }
    return 0;
}
