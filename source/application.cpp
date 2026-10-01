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

        VkPipelineLayout vk_cone_layout = VK_NULL_HANDLE;
        VkPipeline vk_cone_pipeline = VK_NULL_HANDLE;
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

        const VkPipelineShaderStageCreateInfo stages[] = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vk_vertex_shader,
                .pName = "main",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = vk_fragment_shader,
                .pName = "main",
            },
        };

        const VkVertexInputBindingDescription vertex_bindings[1] = {
            {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
            },
        };

        const VkVertexInputAttributeDescription vertex_attributes[2] = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, pos),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color)
            }
        };

        const VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
            .pVertexBindingDescriptions = vertex_bindings,
            .vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
            .pVertexAttributeDescriptions = vertex_attributes,
        };

        const VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        };

        const VkPipelineViewportStateCreateInfo viewport_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        const VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1,
        };

        const VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
        };

        const VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
        };

        const VkPipelineColorBlendAttachmentState color_blend_attachment_state = {
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT
        };

        const VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment_state,
        };

        const VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };

        const VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamic_states,
        };

        const VkPipelineLayoutCreateInfo layout_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
        };

        if (vkCreatePipelineLayout(context.device, &layout_create_info, nullptr, &vk_cone_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout!\n";
            return false;
        }

        const VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = sizeof(stages),
            .pStages = stages,
            .pVertexInputState = &vertex_input_state_create_info,
            .pInputAssemblyState = &input_assembly_state_create_info,
            .pViewportState = &viewport_state_create_info,
            .pRasterizationState = &rasterization_state_create_info,
            .pMultisampleState = &multisample_state_create_info,
            .pDepthStencilState = &depth_stencil_state_create_info,
            .pColorBlendState = &color_blend_state_create_info,
            .pDynamicState = &dynamic_state_create_info,
            .layout = vk_cone_layout,
            .renderPass = context.render_pass,
            .subpass = 0,
        };

        if (vkCreateGraphicsPipelines(
                context.device,
                VK_NULL_HANDLE,
                1,
                &graphics_pipeline_create_info,
                nullptr,
                &vk_cone_pipeline
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline\n";
            return false;
        }

        std::cout << "Pipeline created\n";
        return true;
    }

    void shutdown() {
        auto &context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (vk_cone_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, vk_cone_pipeline, nullptr);
            vk_cone_pipeline = VK_NULL_HANDLE;
        }

        if (vk_cone_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context.device, vk_cone_layout, nullptr);
            vk_cone_layout = VK_NULL_HANDLE;
        }

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
