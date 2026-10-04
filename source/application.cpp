#include "application.hpp"
#include "math.hpp"

#include <array>
#include <imgui.h>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>

namespace application {
    struct GlobalUniforms {
        math::Mat4 model;
        math::Mat4 view;
        math::Mat4 proj;
        float color[4]{};
    };

    struct Vertex {
        float pos[3];
        float color[3];
    };

    std::array<float, 3> color_from_pos(float x, float y, float z) {
        return {
            x * 0.5f + 0.5f,
            y * 0.5f + 0.5f,
            z * 0.5f + 0.5f,
        };
    }

    std::vector<Vertex> make_cone_vertexes(const int n) {
        std::vector<Vertex> array;
        array.reserve(n + 2);

        constexpr float apex_y = 1;
        constexpr float base_y = -1;
        constexpr float radius = 0.6;

        {
            const auto color = color_from_pos(0, apex_y, 0);
            array.push_back(Vertex{{0, apex_y, 0}, {color[0], color[1], color[2]}});
        }

        for (int i = 0; i < n; ++i) {
            const float step = 2 * std::numbers::pi_v<float> / static_cast<float>(n);
            const float corner = static_cast<float>(i) * step;

            const float x = std::cos(corner) * radius;
            const float z = std::sin(corner) * radius;

            const auto color = color_from_pos(x, base_y, z);
            array.push_back(Vertex{{x, base_y, z}, {color[0], color[1], color[2]}});
        }

        {
            const auto color = color_from_pos(0, base_y, 0);
            array.push_back(Vertex{{0, base_y, 0}, {color[0], color[1], color[2]}});
        }

        return array;
    }

    std::vector<uint32_t> make_cone_indexes(int n) {
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
        uint32_t vk_index_count = 0;

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

        VkPipelineLayout vk_cone_layout = VK_NULL_HANDLE;
        VkPipeline vk_cone_pipeline = VK_NULL_HANDLE;

        VkBuffer vk_global_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_global_uniform_buffer_allocation = VK_NULL_HANDLE;
        GlobalUniforms *vk_global_uniform_memory = nullptr;

        VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;

        bool ui_use_perspective = true;
        float ui_fov_deg = 60; // угол обзора по вертикали
        float ui_ortho_size = 1.5;
        float ui_position[3] = {0, 0, 0};
        float ui_rotation[3] = {0, 0, 0}; // в градусах
        float ui_scale[3] = {1, 1, 1};

        bool ui_playing = false;
        float ui_anim_speed = 1;
        float ui_path_radius = 2;
        float ui_path_height = 0.6; // амплитуда по Y
        float ui_anim_time = 0;

        float ui_color[3] = {1.0f, 1.0f, 1.0f};
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
        vk_index_count = indexes.size();
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

        vk_vertex_shader = load_shader_module("../shaders/cone.vert.spv");
        vk_fragment_shader = load_shader_module("../shaders/cone.frag.spv");
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
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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

        const VkDescriptorSetLayoutBinding descriptor_binding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };

        const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &descriptor_binding,
        };

        if (vkCreateDescriptorSetLayout(
                context.device,
                &descriptor_set_layout_info,
                nullptr,
                &vk_descriptor_set_layout
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor set layout\n";
            return false;
        }

        const VkBufferCreateInfo uniform_buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(GlobalUniforms),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        constexpr VmaAllocationCreateInfo uniform_buffer_allocation = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };

        VmaAllocationInfo uniform_allocation_info{};
        if (vmaCreateBuffer(
                context.allocator,
                &uniform_buffer_info,
                &uniform_buffer_allocation,
                &vk_global_uniform_buffer,
                &vk_global_uniform_buffer_allocation,
                &uniform_allocation_info
            ) != VK_SUCCESS) {
            std::cerr << "Failed to create uniform buffer\n";
            return false;
        }
        vk_global_uniform_memory = static_cast<GlobalUniforms *>(uniform_allocation_info.pMappedData);

        const VkPipelineLayoutCreateInfo layout_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(context.device, &layout_create_info, nullptr, &vk_cone_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout!\n";
            return false;
        }

        const VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = sizeof(stages) / sizeof(stages[0]),
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

        const VkDescriptorPoolSize descriptor_pool_size = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        };

        const VkDescriptorPoolCreateInfo descriptor_pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &descriptor_pool_size,
        };

        if (vkCreateDescriptorPool(context.device, &descriptor_pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }


        const VkDescriptorSetAllocateInfo descriptor_set_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(context.device, &descriptor_set_info, &vk_descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set\n";
            return false;
        }

        const VkDescriptorBufferInfo descriptor_buffer_info = {
            .buffer = vk_global_uniform_buffer,
            .offset = 0,
            .range = sizeof(GlobalUniforms),
        };

        const VkWriteDescriptorSet descriptor_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &descriptor_buffer_info,
        };

        vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);
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

        if (vk_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
            vk_descriptor_pool = VK_NULL_HANDLE;
            vk_descriptor_set = VK_NULL_HANDLE;
        }

        if (vk_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);
            vk_descriptor_set_layout = VK_NULL_HANDLE;
        }

        if (vk_global_uniform_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_global_uniform_buffer, vk_global_uniform_buffer_allocation);
            vk_global_uniform_buffer = VK_NULL_HANDLE;
            vk_global_uniform_buffer_allocation = VK_NULL_HANDLE;
            vk_global_uniform_memory = nullptr;
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

    void update(const double time) {
        ImGui::ShowDemoWindow();

        static double last_time = 0;
        const auto delta_time = time - last_time;
        last_time = time;

        ImGui::Begin("Controls");

        ImGui::Checkbox("Perspective projection", &ui_use_perspective);
        if (ui_use_perspective) {
            ImGui::SliderFloat("FOV (deg)", &ui_fov_deg, 10, 120);
        } else {
            ImGui::SliderFloat("Ortho half-height", &ui_ortho_size, 0.5, 10);
        }

        ImGui::Separator();

        ImGui::Checkbox("Play animation", &ui_playing);
        ImGui::SliderFloat("Speed", &ui_anim_speed, 0, 5);
        ImGui::SliderFloat("Path radius", &ui_path_radius, 0, 5);
        ImGui::SliderFloat("Path height", &ui_path_height, 0, 2);

        ImGui::Separator();

        ImGui::BeginDisabled(ui_playing);
        ImGui::SliderFloat3("Position", ui_position, -5, 5);
        ImGui::EndDisabled();

        ImGui::SliderFloat3("Rotation (deg)", ui_rotation, -180, 180);
        ImGui::SliderFloat3("Scale", ui_scale, 0.1, 5);

        ImGui::Separator();
        ImGui::ColorEdit3("Object color", ui_color);

        ImGui::End();

        if (ui_playing) {
            ui_anim_time += static_cast<float>(delta_time) * ui_anim_speed;
        }

        const float t = ui_anim_time;
        const float r = ui_path_radius;
        const float y_amp = ui_path_height;

        const float px = r * std::sin(t);
        const float pz = r * std::sin(t) * std::cos(t) * 2;
        const float py = y_amp * std::sin(t * 2);

        const math::Vec3 animated_pos{px, py, pz};

        const math::Vec3 pos = ui_playing ? animated_pos : math::Vec3{ui_position[0], ui_position[1], ui_position[2]};

        if (ui_playing) {
            ui_position[0] = pos.x;
            ui_position[1] = pos.y;
            ui_position[2] = pos.z;
        }

        const auto &ctx = graphics::internal::context;

        const auto w = static_cast<float>(ctx.swapchain_extent.width);
        const auto h = static_cast<float>(ctx.swapchain_extent.height);
        if (w == 0 || h == 0) return;
        const float aspect = w / h;

        constexpr auto deg2rad = std::numbers::pi_v<float> / 180;

        const math::Mat4 model =
                math::Mat4::translate(pos.x, pos.y, pos.z)
                * math::Mat4::rotate_y(ui_rotation[1] * deg2rad)
                * math::Mat4::rotate_x(ui_rotation[0] * deg2rad)
                * math::Mat4::rotate_z(ui_rotation[2] * deg2rad)
                * math::Mat4::scale(ui_scale[0], ui_scale[1], ui_scale[2]);

        const math::Mat4 view = math::Mat4::lookAt(
            {0, 0, 4},
            {0, 0, 0},
            {0, 1, 0}
        );

        math::Mat4 proj;
        if (ui_use_perspective) {
            proj = math::Mat4::perspective(ui_fov_deg * deg2rad, aspect, 0.1, 100);
        } else {
            const float half_h = ui_ortho_size;
            const float half_w = half_h * aspect;
            // top = -half_h, bottom = +half_h — это даёт тот же Y-flip,
            // что и в perspective (потому что ortho у нас не флипает сам)
            proj = math::Mat4::ortho(-half_w, half_w, half_h, -half_h, 0.1, 100);
        }

        vk_global_uniform_memory->model = model;
        vk_global_uniform_memory->view = view;
        vk_global_uniform_memory->proj = proj;

        vk_global_uniform_memory->color[0] = ui_color[0];
        vk_global_uniform_memory->color[1] = ui_color[1];
        vk_global_uniform_memory->color[2] = ui_color[2];
        vk_global_uniform_memory->color[3] = 1;
    }

    void render(const graphics::internal::FrameData &fd) {
        auto &context = graphics::internal::context;

        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(fd.command_buffer, &begin);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = {0.1, 0.1, 0.1, 0.1}}},
            {.depthStencil = {1, 0}},
        };

        const VkRenderPassBeginInfo render_pass_begin_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.extent = context.swapchain_extent},
            .clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(context.swapchain_extent.width),
            .height = static_cast<float>(context.swapchain_extent.height),
            .minDepth = 0,
            .maxDepth = 1,
        };

        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

        const VkRect2D scissor = {.extent = context.swapchain_extent};
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(
            fd.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_cone_layout,
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_cone_pipeline);

        const VkDeviceSize device_size = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &device_size);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(fd.command_buffer, vk_index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }
} // namespace application
