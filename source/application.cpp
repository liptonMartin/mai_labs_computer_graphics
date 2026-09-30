#include "application.hpp"

#include <imgui.h>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>

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

    VkShaderModule load_shader_module(const std::string &path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file" << path << "!\n";
            return VK_NULL_HANDLE;
        }

        const size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));

        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data()
        };

        VkShaderModule result;
        if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS) {
            std::cerr << "Failed to create shader module!\n";
            return VK_NULL_HANDLE;
        }

        return result;
    }

    namespace {
        constexpr int CONE_SEGMENTS = 24;

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;
    }

    bool initialize() {
        auto &context = graphics::internal::context;

        const auto vertexes = make_cone_vertexes(CONE_SEGMENTS);

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

        VmaAllocationInfo vertex_allocation_info{};
        if (vmaCreateBuffer(
                context.allocator,
                &vertex_buffer,
                &vertex_buffer_allocation,
                &vk_vertex_buffer,
                &vk_vertex_buffer_allocation,
                &vertex_allocation_info
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create and allocate vertex buffer\n";
            return false;
        }

        std::memcpy(vertex_allocation_info.pMappedData, vertexes.data(), sizeof(Vertex) * vertexes.size());
        std::cout << "Successfully created, allocated and mapped vertex buffer memory\n";

        const auto indexes = make_cone_indexes(CONE_SEGMENTS);
        const VkBufferCreateInfo index_buffer = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(uint32_t) * indexes.size(),
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        constexpr VmaAllocationCreateInfo index_buffer_allocation = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };

        VmaAllocationInfo index_allocation_info{};
        if (vmaCreateBuffer(
                context.allocator,
                &index_buffer,
                &index_buffer_allocation,
                &vk_index_buffer,
                &vk_index_buffer_allocation,
                &index_allocation_info
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create and allocate index buffer\n";
            return false;
        }

        std::memcpy(index_allocation_info.pMappedData, indexes.data(), sizeof(uint32_t) * indexes.size());
        std::cout << "Successfully created, allocated and mapped index buffer memory\n";

        vk_vertex_shader = load_shader_module("../cone.vert.spv");
        vk_fragment_shader = load_shader_module("../cone.frag.spv");
        std::cout << "Shaders successfully loaded\n";

        return true;
    }

    void shutdown() {
        auto &context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (vk_vertex_shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);
            vk_vertex_shader = VK_NULL_HANDLE;
        }

        if (vk_fragment_shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);
            vk_fragment_shader = VK_NULL_HANDLE;
        }

        if (vk_vertex_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
            vk_vertex_buffer = VK_NULL_HANDLE;
            vk_vertex_buffer_allocation = VK_NULL_HANDLE;
        }

        if (vk_index_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
            vk_index_buffer = VK_NULL_HANDLE;
            vk_index_buffer_allocation = VK_NULL_HANDLE;
        }
    }

    void update([[maybe_unused]] double time) {
        ImGui::ShowDemoWindow();
    }

    void render(const graphics::internal::FrameData &fd) {
        (void) fd;
    }
} // namespace application
