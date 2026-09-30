#include "application.hpp"

#include <imgui.h>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>
#include <cstring>

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

    namespace {
        constexpr int cone_segments = 24;

        VkBuffer vk_vertex_buffer;
        VmaAllocation vk_vertex_buffer_allocation;
        Vertex *vk_vertex_buffer_memory;
    }

    bool initialize() {
        auto &context = graphics::internal::context;

        const auto vertexes = make_cone_vertexes(cone_segments);

        const VkBufferCreateInfo vertex_buffer = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(Vertex) * vertexes.size(),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        constexpr VmaAllocationCreateInfo vertex_buffer_allocation = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };

        if (vmaCreateBuffer(
                context.allocator,
                &vertex_buffer,
                &vertex_buffer_allocation,
                &vk_vertex_buffer,
                &vk_vertex_buffer_allocation,
                nullptr
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create and allocate vertex buffer\n";
            return false;
        }

        if (vmaMapMemory(
                context.allocator, vk_vertex_buffer_allocation, reinterpret_cast<void **>(&vk_vertex_buffer_memory)
            ) != VK_SUCCESS) {
            std::cerr << "Failed to map vertex buffer memory\n";
            return false;
        }

        memcpy(vk_vertex_buffer_memory, vertexes.data(), sizeof(Vertex) * vertexes.size());
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
