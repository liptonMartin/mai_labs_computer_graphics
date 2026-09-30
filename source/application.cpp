#include "application.hpp"

#include <imgui.h>
#include <vector>
#include <numbers>
#include <cmath>

namespace application {
    struct Vertex {
        float pos[3];
        float color[3];
    };

    static std::vector<Vertex> make_cone_vertexes(int n) {
        std::vector<Vertex> array;
        array.reserve(n + 2);

        Vertex apex{{0, -0.8, 0}, {1, 0, 0}};
        Vertex center{{0, 0.5, 0}, {0, 1, 0}};
        array.push_back(apex);

        constexpr float radius = 0.6;
        for (int i = 0; i < n; ++i) {
            const float step = 2 * std::numbers::pi_v<float> / static_cast<float>(n);
            const float corner = static_cast<float>(i) * step;

            const float x = std::cos(corner) * radius;
            const float y = 0.5;
            const float z = std::sin(corner) * radius;

            Vertex vertex{{x, y, z}, {0, 0, 1}};
            array.push_back(vertex);
        }
        array.push_back(center);
        return array;
    }

    static std::vector<uint32_t> make_cone_indexes(int n) {
        std::vector<uint32_t> indexes;
        indexes.reserve(n * 6);

        const uint32_t apex = 0;
        const uint32_t center = n + 1;
        for (int i = 1; i <= n; ++i) {
            const uint32_t current = i;
            const uint32_t next = 1 + i % n;

            indexes.insert(indexes.end(), {apex, current, next});
            indexes.insert(indexes.end(), {center, current, next});
        }

        return indexes;
    }

    bool initialize() {
        return true;
    }

    void shutdown() {
        auto &context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
    }

    void update([[maybe_unused]] double time) {
        ImGui::ShowDemoWindow();
    }

    void render(const graphics::internal::FrameData &fd) {
        (void) fd;
    }
} // namespace application
