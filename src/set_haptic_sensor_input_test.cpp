#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HapticGlove.hpp>
#include <SenseGlove/Core/Library.hpp>
#include <SenseGlove/Core/Nova2Glove.hpp>
#include <SenseGlove/Core/SenseCom.hpp>

using namespace SGCore;

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    constexpr std::chrono::milliseconds kUpdateInterval{50};
    constexpr float kFallbackMinForceN = 0.0f;
    constexpr float kFallbackMaxForceN = 0.0f;
    constexpr double kDemoFrequencyHz = 0.2; // Used only for the demo waveform

    struct ForceCalibrationSample
    {
        float forceN;
        float level;
    };

    // Replace these with your measured level↔force pairs when available.
    constexpr std::array<ForceCalibrationSample, 6> kCalibrationTable{{
        {0.0f, 0.0f},
        {3.0f, 0.25f},
        {5.0f, 0.4f},
        {8.0f, 0.6f},
        {12.0f, 0.85f},
        {18.0f, 1.0f},
    }};

    void PrintLibraryInfo()
    {
        std::cout << "SenseGlove C++ Library: " << Library::Version()
                  << " | Backend: " << Library::BackendVersion()
                  << " | SGConnect: " << Library::SGConnectVersion() << std::endl;
    }

    void EnsureSenseCom()
    {
        if (SenseCom::ScanningActive()) {
            return;
        }

        std::cout << "Starting SenseCom for device discovery..." << std::endl;
        if (SenseCom::StartupSenseCom()) {
            std::cout << "SenseCom started; waiting a moment for devices." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        } else {
            std::cout << "SenseCom startup failed or was already running." << std::endl;
        }
    }

    std::shared_ptr<Nova::Nova2Glove> GetAnyNova2()
    {
        std::shared_ptr<HapticGlove> glove;
        if (HandLayer::GetGloveInstance(true, glove)) {
            return std::dynamic_pointer_cast<Nova::Nova2Glove>(glove);
        }

        if (HandLayer::GetGloveInstance(false, glove)) {
            return std::dynamic_pointer_cast<Nova::Nova2Glove>(glove);
        }

        return nullptr;
    }

    float NewtonsToLevel(float newtons)
    {
        if (kCalibrationTable.empty()) {
            if (kFallbackMaxForceN <= 0.0f) {
                return 0.0f;
            }
            const float normalized = newtons / kFallbackMaxForceN;
            return std::clamp(normalized, 0.0f, 1.0f);
        }

        if (newtons <= kCalibrationTable.front().forceN) {
            return kCalibrationTable.front().level;
        }
        if (newtons >= kCalibrationTable.back().forceN) {
            return kCalibrationTable.back().level;
        }

        for (size_t i = 1; i < kCalibrationTable.size(); ++i) {
            if (newtons <= kCalibrationTable[i].forceN) {
                const auto& a = kCalibrationTable[i - 1];
                const auto& b = kCalibrationTable[i];
                const float span = b.forceN - a.forceN;
                const float t = span > 0.0f ? (newtons - a.forceN) / span : 0.0f;
                return a.level + t * (b.level - a.level);
            }
        }

        return kCalibrationTable.back().level;
    }

    std::vector<float> AcquireSensorForces(double elapsedSeconds)
    {
        // Replace this section with the actual F/T or retargeting feed.
        const float offset = (kFallbackMaxForceN + kFallbackMinForceN) * 0.5f;
        const float amplitude = (kFallbackMaxForceN - kFallbackMinForceN) * 0.5f;
        const double radians = 2.0 * kPi * kDemoFrequencyHz * elapsedSeconds;
        const float sample = offset + amplitude * static_cast<float>(std::sin(radians));
        return std::vector<float>(4, sample);
    }

    void SendForceLevels(Nova::Nova2Glove& glove, const std::vector<float>& fingertipForcesN)
    {
        constexpr size_t kChannels = 4;
        std::vector<float> levels(kChannels, 0.0f);

        const size_t count = std::min(kChannels, fingertipForcesN.size());
        for (size_t i = 0; i < count; ++i) {
            levels[i] = NewtonsToLevel(std::max(fingertipForcesN[i], 0.0f));
        }

        if (glove.QueueForceFeedbackLevels(levels) && glove.SendHaptics()) {
            std::cout << "Target fingertip forces (N): ";
            for (size_t i = 0; i < count; ++i) {
                std::cout << fingertipForcesN[i] << (i + 1 < count ? ", " : "");
            }
            std::cout << '\r';
            std::cout.flush();
            return;
        }

        std::cout << "\nFailed to queue/send force levels." << std::endl;
    }
}

int main()
{
    PrintLibraryInfo();
    EnsureSenseCom();

    auto glove = GetAnyNova2();
    if (!glove) {
        std::cout << "No Nova2 glove detected." << std::endl;
        return 0;
    }

    std::cout << "Streaming fingertip forces from sensor input (demo uses sine feed)." << std::endl;

    const auto startTime = std::chrono::steady_clock::now();

    while (HandLayer::GlovesConnected() > 0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(now - startTime).count();
        const auto desiredForces = AcquireSensorForces(elapsedSeconds);

        SendForceLevels(*glove, desiredForces);
        std::this_thread::sleep_for(kUpdateInterval);
    }

    std::cout << "No more gloves connected; exiting." << std::endl;
    return 0;
}
