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
#include <map>
#include <ranges>

namespace application {
    enum class Figure {
        Cube,
        Cone,
    };

    struct DirectionalLightUniform {
        float direction[3]; // offset 0
        float intensity; // offset 12
        float color[3]; // offset 16
        float _padding; // offset 28  -> размер 32
    };

    static_assert(sizeof(DirectionalLightUniform) == 32);

    struct PointLightUniform {
        float position[3]; // offset 0
        float intensity; // offset 12
        float color[3]; // offset 16
        float radius; // offset 28
        // size = 32, align = 16
    };

    static_assert(sizeof(PointLightUniform) == 32);

    struct SpotLightUniform {
        float position[3]; // offset 0
        float _pad0; // offset 12
        float direction[3]; // offset 16
        float intensity; // offset 28
        float color[3]; // offset 32
        float innerAngle; // offset 44
        float outerAngle; // offset 48
        float radius; // offset 52
        float _pad1[2]; // offset 56..63
        // size = 64, align = 16
    };

    static_assert(sizeof(SpotLightUniform) == 64);

    struct SceneUniforms {
        math::Mat4 view; // offset 0
        math::Mat4 proj; // offset 64
        float cameraPos[3]; // offset 128
        float ambientStrength; // offset 140
        float ambientColor[3]; // offset 144
        float _padding0; // offset 156
        DirectionalLightUniform directional; // offset 160
        uint32_t pointLightCount; // offset 192
        uint32_t spotLightCount; // offset 196
        float _padding1[2]; // offset 200..208
    };

    static_assert(sizeof(SceneUniforms) == 208);

    struct ModelUniforms {
        math::Mat4 model; // offset 0
        float albedo[3]; // offset 64
        float shininess; // offset 76
        float specularColor[3]; // offset 80
        float _padding; // offset 92  -> размер 96
    };

    static_assert(sizeof(ModelUniforms) == 96);

    struct Vertex {
        float pos[3];
        float color[3];
        float normal[3];
    };

    struct Object {
        uint32_t vk_index_count{};

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        // model uniforms (set=1)
        VkBuffer vk_model_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
        ModelUniforms *vk_model_uniform_memory = nullptr;
        VkDescriptorSet vk_model_descriptor_set = VK_NULL_HANDLE;

        Object() = default;

        bool create_vertex_and_index_buffers(
            const graphics::internal::Context &context,
            const std::vector<Vertex> &vertexes,
            const std::vector<uint32_t> &indexes
        ) {
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
            return true;
        }

        bool create_model_resources(
            const graphics::internal::Context &context,
            const VkDescriptorPool &vk_descriptor_pool,
            const VkDescriptorSetLayout &vk_model_descriptor_set_layout
        ) {
            // --- model uniform buffer ---
            const VkBufferCreateInfo uniform_buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(ModelUniforms),
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
                    &vk_model_uniform_buffer,
                    &vk_model_uniform_buffer_allocation,
                    &uniform_allocation_info
                ) != VK_SUCCESS) {
                std::cerr << "Failed to create model uniform buffer\n";
                return false;
            }
            vk_model_uniform_memory = static_cast<ModelUniforms *>(uniform_allocation_info.pMappedData);

            // --- descriptor set ---
            const VkDescriptorSetAllocateInfo descriptor_set_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_model_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(context.device, &descriptor_set_info, &vk_model_descriptor_set) !=
                VK_SUCCESS) {
                std::cerr << "Failed to allocate model descriptor set\n";
                return false;
            }

            const VkDescriptorBufferInfo descriptor_buffer_info = {
                .buffer = vk_model_uniform_buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            };

            const VkWriteDescriptorSet descriptor_write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_model_descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &descriptor_buffer_info,
            };

            vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);
            std::cout << "Successfully created model uniform memory and descriptor set\n";
            return true;
        }
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
        array.reserve(2 * n + 2);

        constexpr float apex_y = 1;
        constexpr float base_y = -1;
        constexpr float radius = 0.6f;
        const float height = apex_y - base_y; // = 2

        // 0: вершина конуса
        {
            const auto color = color_from_pos(0, apex_y, 0);
            array.push_back(Vertex{
                {0, apex_y, 0},
                {color[0], color[1], color[2]},
                {0, 1, 0}
            });
        }

        // 1..n: боковая поверхность (вершины на окружности основания)
        for (int i = 0; i < n; ++i) {
            const float step = 2 * std::numbers::pi_v<float> / static_cast<float>(n);
            const float corner = static_cast<float>(i) * step;

            const float x = std::cos(corner) * radius;
            const float z = std::sin(corner) * radius;

            const float nx = height * std::cos(corner);
            const float ny = radius;
            const float nz = height * std::sin(corner);
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);

            const auto color = color_from_pos(x, base_y, z);
            array.push_back(Vertex{
                {x, base_y, z},
                {color[0], color[1], color[2]},
                {nx / len, ny / len, nz / len}
            });
        }

        // n+1: центр основания
        {
            const auto color = color_from_pos(0, base_y, 0);
            array.push_back(Vertex{
                {0, base_y, 0},
                {color[0], color[1], color[2]},
                {0, -1, 0}
            });
        }

        // n+2..2n+1: дубли вершин окружности для плоского основания
        for (int i = 0; i < n; ++i) {
            const float step = 2 * std::numbers::pi_v<float> / static_cast<float>(n);
            const float corner = static_cast<float>(i) * step;

            const float x = std::cos(corner) * radius;
            const float z = std::sin(corner) * radius;

            const auto color = color_from_pos(x, base_y, z);
            array.push_back(Vertex{
                {x, base_y, z},
                {color[0], color[1], color[2]},
                {0, -1, 0}
            });
        }

        return array;
    }

    std::vector<uint32_t> make_cone_indexes(int n) {
        std::vector<uint32_t> indexes;
        indexes.reserve(n * 6);

        const uint32_t apex = 0;
        const uint32_t base_center = n + 1;

        for (int i = 0; i < n; ++i) {
            const uint32_t current = 1 + i;
            const uint32_t next = 1 + (i + 1) % n;
            indexes.insert(indexes.end(), {apex, current, next});

            const uint32_t base_current = n + 2 + i;
            const uint32_t base_next = n + 2 + (i + 1) % n;
            indexes.insert(indexes.end(), {base_center, base_current, base_next});
        }

        return indexes;
    }

    std::vector<Vertex> make_cube_vertexes() {
        constexpr float s = 0.3f;
        const std::array<float, 3> red = {1, 0, 0};
        const std::array<float, 3> green = {0, 1, 0};

        std::vector<Vertex> array;
        array.reserve(24);

        auto push = [&](float x, float y, float z,
                        const std::array<float, 3> &c,
                        const std::array<float, 3> &n) {
            array.push_back(Vertex{{x, y, z}, {c[0], c[1], c[2]}, {n[0], n[1], n[2]}});
        };

        // FRONT (+Z)
        push(s, s, s, red, {0, 0, 1}); // 0 TR
        push(-s, s, s, red, {0, 0, 1}); // 1 TL
        push(-s, -s, s, green, {0, 0, 1}); // 2 BL
        push(s, -s, s, green, {0, 0, 1}); // 3 BR

        // BACK (-Z)
        push(-s, s, -s, red, {0, 0, -1}); // 4
        push(s, s, -s, red, {0, 0, -1}); // 5
        push(s, -s, -s, green, {0, 0, -1}); // 6
        push(-s, -s, -s, green, {0, 0, -1}); // 7

        // RIGHT (+X)
        push(s, s, -s, red, {1, 0, 0}); // 8
        push(s, s, s, red, {1, 0, 0}); // 9
        push(s, -s, s, green, {1, 0, 0}); // 10
        push(s, -s, -s, green, {1, 0, 0}); // 11

        // LEFT (-X)
        push(-s, s, s, red, {-1, 0, 0}); // 12
        push(-s, s, -s, red, {-1, 0, 0}); // 13
        push(-s, -s, -s, green, {-1, 0, 0}); // 14
        push(-s, -s, s, green, {-1, 0, 0}); // 15

        // TOP (+Y)
        push(s, s, -s, red, {0, 1, 0}); // 16
        push(-s, s, -s, red, {0, 1, 0}); // 17
        push(-s, s, s, red, {0, 1, 0}); // 18
        push(s, s, s, red, {0, 1, 0}); // 19

        // BOTTOM (-Y)
        push(s, -s, s, green, {0, -1, 0}); // 20
        push(-s, -s, s, green, {0, -1, 0}); // 21
        push(-s, -s, -s, green, {0, -1, 0}); // 22
        push(s, -s, -s, green, {0, -1, 0}); // 23

        return array;
    }

    std::vector<uint32_t> make_cube_indexes() {
        std::vector<uint32_t> indexes;
        indexes.reserve(6 * 2 * 3);

        for (uint32_t face = 0; face < 6; ++face) {
            const uint32_t b = face * 4;
            indexes.insert(indexes.end(), {b + 0, b + 2, b + 1, b + 0, b + 3, b + 2});
        }

        return indexes;
    }

    math::Vec3 create_animated_position_cone(const float time, const float radius, const float y_amplitude) {
        const float px = radius * std::sin(time);
        const float pz = radius * std::sin(time) * std::cos(time) * 2;
        const float py = y_amplitude * std::sin(time * 2);

        return {px, py, pz};
    }

    math::Vec3 create_animated_position_cube(const float time, const float radius) {
        const float px = radius * std::sin(time);
        const float pz = radius * std::cos(time);
        const float py = 0;

        return {px, py, pz};
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

        std::map<Figure, Object> objects{};

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

        VkPipelineLayout vk_cone_layout = VK_NULL_HANDLE;
        VkPipeline vk_cone_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;

        VkBuffer vk_scene_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_scene_uniform_buffer_allocation = VK_NULL_HANDLE;
        SceneUniforms *vk_scene_uniform_memory = nullptr;
        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;

        std::vector<PointLightUniform> point_lights{};
        std::vector<SpotLightUniform> spot_lights{};

        struct LightStorageBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            void *mapped = nullptr;
            uint32_t capacity = 0;
        };

        LightStorageBuffer point_lights_buffer{};
        LightStorageBuffer spot_lights_buffer{};

        bool recreate_light_buffer(
            const graphics::internal::Context &ctx,
            LightStorageBuffer &storage,
            uint32_t new_capacity,
            size_t element_size,
            VkDescriptorSet dst_set,
            uint32_t dst_binding
        ) {
            if (storage.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(ctx.allocator, storage.buffer, storage.allocation);
                storage.buffer = VK_NULL_HANDLE;
                storage.allocation = VK_NULL_HANDLE;
                storage.mapped = nullptr;
            }

            const VkBufferCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = element_size * new_capacity,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            constexpr VmaAllocationCreateInfo alloc = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            VmaAllocationInfo out_info{};
            if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                                &storage.buffer, &storage.allocation, &out_info) != VK_SUCCESS) {
                std::cerr << "Failed to create light storage buffer\n";
                return false;
            }
            storage.mapped = out_info.pMappedData;
            storage.capacity = new_capacity;

            std::memset(storage.mapped, 0, element_size * new_capacity);

            const VkDescriptorBufferInfo buf_info = {
                .buffer = storage.buffer,
                .offset = 0,
                .range = element_size * new_capacity,
            };
            const VkWriteDescriptorSet write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = dst_set,
                .dstBinding = dst_binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buf_info,
            };
            vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
            return true;
        }

        bool ui_use_perspective = true;
        float ui_fov_deg = 60;
        float ui_ortho_size = 1.5;
        float ui_camera_position[3] = {0, 0, 4};
        float ui_camera_yaw_deg = 0;
        float ui_camera_pitch_deg = 0;
        float ui_camera_near = 0.1f;
        float ui_camera_far = 100.0f;

        float ui_position[3] = {0, 0, 0};
        float ui_rotation[3] = {0, 0, 0};
        float ui_scale[3] = {1, 1, 1};

        bool ui_playing = false;
        float ui_anim_speed = 1;
        float ui_path_radius = 2;
        float ui_path_height = 0.6;
        float ui_anim_time = 0;

        float ui_ambient_strength = 0.15f;
        float ui_ambient_color[3] = {1, 1, 1};

        float ui_dir_direction[3] = {0.4f, -1.0f, -0.5f};
        float ui_dir_color[3] = {1, 1, 1};
        float ui_dir_intensity = 0;

        float ui_albedo[3] = {1, 1, 1};
        float ui_specular_color[3] = {1, 1, 1};
        float ui_shininess = 32.0f;
    }

    bool initialize() {
        auto &context = graphics::internal::context;

        const auto cone_vertexes = make_cone_vertexes(CONE_SEGMENTS);
        const auto cone_indexes = make_cone_indexes(CONE_SEGMENTS);

        const auto cube_vertexes = make_cube_vertexes();
        const auto cube_indexes = make_cube_indexes();

        auto cone = Object();
        if (!cone.create_vertex_and_index_buffers(context, cone_vertexes, cone_indexes))
            return false;

        auto cube = Object();
        if (!cube.create_vertex_and_index_buffers(context, cube_vertexes, cube_indexes))
            return false;

        objects[Figure::Cone] = cone;
        objects[Figure::Cube] = cube;

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

        const VkVertexInputAttributeDescription vertex_attributes[3] = {
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
                .offset = offsetof(Vertex, color),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
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
            .cullMode = VK_CULL_MODE_BACK_BIT,
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

        const VkDescriptorSetLayoutBinding scene_bindings[3] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        };
        const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = scene_bindings,
        };

        if (vkCreateDescriptorSetLayout(
                context.device, &scene_layout_info, nullptr,
                &vk_scene_descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create scene descriptor set layout\n";
            return false;
        }

        const VkDescriptorSetLayoutBinding model_binding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const VkDescriptorSetLayoutCreateInfo model_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &model_binding,
        };
        if (vkCreateDescriptorSetLayout(
                context.device, &model_layout_info, nullptr,
                &vk_model_descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create model descriptor set layout\n";
            return false;
        }

        const VkDescriptorSetLayout set_layouts[2] = {
            vk_scene_descriptor_set_layout,
            vk_model_descriptor_set_layout,
        };
        const VkPipelineLayoutCreateInfo layout_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 2,
            .pSetLayouts = set_layouts,
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
                context.device, VK_NULL_HANDLE, 1,
                &graphics_pipeline_create_info, nullptr,
                &vk_cone_pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline\n";
            return false;
        }
        std::cout << "Pipeline created\n";

        const VkDescriptorPoolSize pool_sizes[] = {
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 3},
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
        };
        const VkDescriptorPoolCreateInfo descriptor_pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 3,
            .poolSizeCount = 2,
            .pPoolSizes = pool_sizes,
        };
        if (vkCreateDescriptorPool(context.device, &descriptor_pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }

        {
            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(SceneUniforms),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            constexpr VmaAllocationCreateInfo allocation_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            VmaAllocationInfo out_info{};
            if (vmaCreateBuffer(
                    context.allocator, &buffer_info, &allocation_info,
                    &vk_scene_uniform_buffer, &vk_scene_uniform_buffer_allocation,
                    &out_info) != VK_SUCCESS) {
                std::cerr << "Failed to create scene uniform buffer\n";
                return false;
            }
            vk_scene_uniform_memory = static_cast<SceneUniforms *>(out_info.pMappedData);
            *vk_scene_uniform_memory = SceneUniforms{};

            const VkDescriptorSetAllocateInfo descriptor_set_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
            };
            if (vkAllocateDescriptorSets(context.device, &descriptor_set_info, &vk_scene_descriptor_set) !=
                VK_SUCCESS) {
                std::cerr << "Failed to allocate scene descriptor set\n";
                return false;
            }

            const VkDescriptorBufferInfo descriptor_buffer_info = {
                .buffer = vk_scene_uniform_buffer,
                .offset = 0,
                .range = sizeof(SceneUniforms),
            };
            const VkWriteDescriptorSet descriptor_write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_scene_descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &descriptor_buffer_info,
            };
            vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);

            if (!recreate_light_buffer(context, point_lights_buffer, 1,
                                       sizeof(PointLightUniform),
                                       vk_scene_descriptor_set, 1))
                return false;
            if (!recreate_light_buffer(context, spot_lights_buffer, 1,
                                       sizeof(SpotLightUniform),
                                       vk_scene_descriptor_set, 2))
                return false;
        }

        for (auto &object: objects | std::views::values) {
            if (!object.create_model_resources(context, vk_descriptor_pool, vk_model_descriptor_set_layout))
                return false;
        }

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

            vk_scene_descriptor_set = VK_NULL_HANDLE;
            for (auto &object: objects | std::views::values)
                object.vk_model_descriptor_set = VK_NULL_HANDLE;
        }

        if (vk_scene_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);
            vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        }
        if (vk_model_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);
            vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        }

        if (vk_scene_uniform_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_scene_uniform_buffer, vk_scene_uniform_buffer_allocation);
            vk_scene_uniform_buffer = VK_NULL_HANDLE;
            vk_scene_uniform_buffer_allocation = VK_NULL_HANDLE;
            vk_scene_uniform_memory = nullptr;
        }

        if (point_lights_buffer.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, point_lights_buffer.buffer, point_lights_buffer.allocation);
            point_lights_buffer = {};
        }
        if (spot_lights_buffer.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, spot_lights_buffer.buffer, spot_lights_buffer.allocation);
            spot_lights_buffer = {};
        }

        for (auto &object: objects | std::views::values) {
            if (object.vk_model_uniform_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(
                    context.allocator,
                    object.vk_model_uniform_buffer,
                    object.vk_model_uniform_buffer_allocation
                );
                object.vk_model_uniform_buffer = VK_NULL_HANDLE;
                object.vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
                object.vk_model_uniform_memory = nullptr;
            }
        }

        for (auto &object: objects | std::views::values) {
            if (object.vk_vertex_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(context.allocator, object.vk_vertex_buffer, object.vk_vertex_buffer_allocation);
                object.vk_vertex_buffer = VK_NULL_HANDLE;
                object.vk_vertex_buffer_allocation = VK_NULL_HANDLE;
            }

            if (object.vk_index_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(context.allocator, object.vk_index_buffer, object.vk_index_buffer_allocation);
                object.vk_index_buffer = VK_NULL_HANDLE;
                object.vk_index_buffer_allocation = VK_NULL_HANDLE;
            }
        }
    }

    void update(const double time) {
        ImGui::ShowDemoWindow();

        static double last_time = 0;
        const auto delta_time = time - last_time;
        last_time = time;

        ImGui::Begin("Controls");

        ImGui::Text("Projection");
        ImGui::Checkbox("Perspective projection", &ui_use_perspective);
        if (ui_use_perspective) {
            ImGui::SliderFloat("FOV (deg)", &ui_fov_deg, 10, 120);
        } else {
            ImGui::SliderFloat("Ortho half-height", &ui_ortho_size, 0.5, 10);
        }
        ImGui::SliderFloat("Near", &ui_camera_near, 0.01f, 5.0f);
        ImGui::SliderFloat("Far", &ui_camera_far, 10.0f, 500.0f);

        ImGui::Separator();
        ImGui::Text("Camera");
        ImGui::SliderFloat3("Camera position", ui_camera_position, -10, 10);
        ImGui::SliderFloat("Camera yaw (deg)", &ui_camera_yaw_deg, -180, 180);
        ImGui::SliderFloat("Camera pitch (deg)", &ui_camera_pitch_deg, -89, 89);

        ImGui::Separator();
        ImGui::Text("Scene lighting");
        ImGui::SliderFloat("Ambient strength", &ui_ambient_strength, 0, 1);
        ImGui::ColorEdit3("Ambient color", ui_ambient_color);
        ImGui::SliderFloat3("Dir light direction", ui_dir_direction, -1, 1);
        ImGui::ColorEdit3("Dir light color", ui_dir_color);
        ImGui::SliderFloat("Dir light intensity", &ui_dir_intensity, 0, 5);

        ImGui::Separator();
        ImGui::Text("Point lights: %zu", point_lights.size());
        if (ImGui::Button("Add point light")) {
            PointLightUniform pl{};
            pl.position[0] = 0.0f;
            pl.position[1] = 1.0f;
            pl.position[2] = 0.0f;
            pl.color[0] = 1.0f;
            pl.color[1] = 0.4f;
            pl.color[2] = 0.2f;
            pl.intensity = 1.0f;
            pl.radius = 4.0f;
            point_lights.push_back(pl);
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove last point light")) {
            if (!point_lights.empty()) point_lights.pop_back();
        }
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i) + 1000);
            if (ImGui::TreeNode("pl", "Point light #%d", static_cast<int>(i))) {
                ImGui::SliderFloat3("Position", point_lights[i].position, -10.0f, 10.0f);
                ImGui::ColorEdit3("Color", point_lights[i].color);
                ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 10.0f);
                ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.0f, 20.0f);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Text("Spot lights: %zu", spot_lights.size());
        if (ImGui::Button("Add spot light")) {
            SpotLightUniform sl{};
            sl.position[0] = 0.0f;
            sl.position[1] = 3.0f;
            sl.position[2] = 0.0f;
            sl.direction[0] = 0.0f;
            sl.direction[1] = -1.0f;
            sl.direction[2] = 0.0f;
            sl.color[0] = 1.0f;
            sl.color[1] = 1.0f;
            sl.color[2] = 1.0f;
            sl.intensity = 2.0f;
            sl.innerAngle = 0.30f; // ~17°
            sl.outerAngle = 0.50f; // ~28°
            sl.radius = 6.0f;
            spot_lights.push_back(sl);
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove last spot light")) {
            if (!spot_lights.empty()) spot_lights.pop_back();
        }
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i) + 2000);
            if (ImGui::TreeNode("sl", "Spot light #%d", static_cast<int>(i))) {
                ImGui::SliderFloat3("Position", spot_lights[i].position, -10.0f, 10.0f);
                ImGui::SliderFloat3("Direction", spot_lights[i].direction, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", spot_lights[i].color);
                ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 10.0f);
                ImGui::SliderFloat("Inner angle", &spot_lights[i].innerAngle, 0.0f, 1.55f);
                ImGui::SliderFloat("Outer angle", &spot_lights[i].outerAngle, 0.0f, 1.55f);
                ImGui::SliderFloat("Radius", &spot_lights[i].radius, 0.0f, 20.0f);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Text("Material");
        ImGui::ColorEdit3("Albedo", ui_albedo);
        ImGui::ColorEdit3("Specular color", ui_specular_color);
        ImGui::SliderFloat("Shininess", &ui_shininess, 1, 256);

        ImGui::Separator();
        ImGui::Text("Animation");
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

        ImGui::End();

        if (ui_playing) {
            ui_anim_time += static_cast<float>(delta_time) * ui_anim_speed;
        }

        const float t = ui_anim_time;
        const float r = ui_path_radius;
        const float y_amp = ui_path_height;

        const auto animated_pos_cone = create_animated_position_cone(t, r, y_amp);
        const auto pos_cube = create_animated_position_cube(t, r);

        const math::Vec3 pos_cone = ui_playing
                                        ? animated_pos_cone
                                        : math::Vec3{ui_position[0], ui_position[1], ui_position[2]};

        if (ui_playing) {
            ui_position[0] = pos_cone.x;
            ui_position[1] = pos_cone.y;
            ui_position[2] = pos_cone.z;
        }

        const auto &ctx = graphics::internal::context;

        const auto w = static_cast<float>(ctx.swapchain_extent.width);
        const auto h = static_cast<float>(ctx.swapchain_extent.height);
        if (w == 0 || h == 0) return;
        const float aspect = w / h;

        constexpr auto deg2rad = std::numbers::pi_v<float> / 180;

        const math::Mat4 cone_model =
                math::Mat4::translate(pos_cone.x, pos_cone.y, pos_cone.z)
                * math::Mat4::rotate_y(ui_rotation[1] * deg2rad)
                * math::Mat4::rotate_x(ui_rotation[0] * deg2rad)
                * math::Mat4::rotate_z(ui_rotation[2] * deg2rad)
                * math::Mat4::scale(ui_scale[0], ui_scale[1], ui_scale[2]);

        const math::Mat4 cube_model = math::Mat4::translate(pos_cube.x, pos_cube.y, pos_cube.z);

        const float yaw_rad = ui_camera_yaw_deg * deg2rad;
        const float pitch_rad = ui_camera_pitch_deg * deg2rad;

        const float cos_pitch = std::cos(pitch_rad);
        const float sin_pitch = std::sin(pitch_rad);
        const float cos_yaw = std::cos(yaw_rad);
        const float sin_yaw = std::sin(yaw_rad);

        const math::Vec3 forward = {
            sin_yaw * cos_pitch,
            sin_pitch,
            -cos_yaw * cos_pitch,
        };

        const math::Vec3 camera_pos = {
            ui_camera_position[0],
            ui_camera_position[1],
            ui_camera_position[2],
        };
        const math::Vec3 target = {
            camera_pos.x + forward.x,
            camera_pos.y + forward.y,
            camera_pos.z + forward.z,
        };

        const math::Mat4 view = math::Mat4::lookAt(camera_pos, target, {0, 1, 0});

        math::Mat4 proj;
        if (ui_use_perspective) {
            proj = math::Mat4::perspective(
                ui_fov_deg * deg2rad, aspect,
                ui_camera_near, ui_camera_far
            );
        } else {
            const float half_h = ui_ortho_size;
            const float half_w = half_h * aspect;
            proj = math::Mat4::ortho(
                -half_w, half_w,
                half_h, -half_h,
                ui_camera_near, ui_camera_far
            );
        }

        vk_scene_uniform_memory->view = view;
        vk_scene_uniform_memory->proj = proj;

        vk_scene_uniform_memory->cameraPos[0] = camera_pos.x;
        vk_scene_uniform_memory->cameraPos[1] = camera_pos.y;
        vk_scene_uniform_memory->cameraPos[2] = camera_pos.z;

        vk_scene_uniform_memory->ambientStrength = ui_ambient_strength;
        vk_scene_uniform_memory->ambientColor[0] = ui_ambient_color[0];
        vk_scene_uniform_memory->ambientColor[1] = ui_ambient_color[1];
        vk_scene_uniform_memory->ambientColor[2] = ui_ambient_color[2];
        vk_scene_uniform_memory->_padding0 = 0.0f;

        vk_scene_uniform_memory->directional.direction[0] = ui_dir_direction[0];
        vk_scene_uniform_memory->directional.direction[1] = ui_dir_direction[1];
        vk_scene_uniform_memory->directional.direction[2] = ui_dir_direction[2];
        vk_scene_uniform_memory->directional.intensity = ui_dir_intensity;
        vk_scene_uniform_memory->directional.color[0] = ui_dir_color[0];
        vk_scene_uniform_memory->directional.color[1] = ui_dir_color[1];
        vk_scene_uniform_memory->directional.color[2] = ui_dir_color[2];
        vk_scene_uniform_memory->directional._padding = 0.0f;

        const auto &ctx2 = graphics::internal::context;

        {
            const uint32_t required = static_cast<uint32_t>(point_lights.size());
            const uint32_t want_capacity = std::max<uint32_t>(1u, required);
            if (want_capacity > point_lights_buffer.capacity) {
                if (!recreate_light_buffer(ctx2, point_lights_buffer, want_capacity,
                                           sizeof(PointLightUniform),
                                           vk_scene_descriptor_set, 1))
                    return;
            }
            if (required > 0) {
                std::memcpy(point_lights_buffer.mapped, point_lights.data(),
                            sizeof(PointLightUniform) * required);
            }
            vk_scene_uniform_memory->pointLightCount = required;
        }

        {
            const uint32_t required = static_cast<uint32_t>(spot_lights.size());
            const uint32_t want_capacity = std::max<uint32_t>(1u, required);
            if (want_capacity > spot_lights_buffer.capacity) {
                if (!recreate_light_buffer(ctx2, spot_lights_buffer, want_capacity,
                                           sizeof(SpotLightUniform),
                                           vk_scene_descriptor_set, 2))
                    return;
            }
            if (required > 0) {
                std::memcpy(spot_lights_buffer.mapped, spot_lights.data(),
                            sizeof(SpotLightUniform) * required);
            }
            vk_scene_uniform_memory->spotLightCount = required;
        }

        {
            auto &mm = objects[Figure::Cone].vk_model_uniform_memory;
            mm->model = cone_model;
            mm->albedo[0] = ui_albedo[0];
            mm->albedo[1] = ui_albedo[1];
            mm->albedo[2] = ui_albedo[2];
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding = 0.0f;
        }
        {
            auto &mm = objects[Figure::Cube].vk_model_uniform_memory;
            mm->model = cube_model;
            mm->albedo[0] = ui_albedo[0];
            mm->albedo[1] = ui_albedo[1];
            mm->albedo[2] = ui_albedo[2];
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding = 0.0f;
        }
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

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_cone_pipeline);

        for (const auto &object: objects | std::views::values) {
            const VkDeviceSize device_size = 0;
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.vk_vertex_buffer, &device_size);
            vkCmdBindIndexBuffer(fd.command_buffer, object.vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

            const VkDescriptorSet descriptor_sets[2] = {
                vk_scene_descriptor_set,
                object.vk_model_descriptor_set,
            };
            vkCmdBindDescriptorSets(
                fd.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_cone_layout,
                0,
                2,
                descriptor_sets,
                0,
                nullptr
            );

            vkCmdDrawIndexed(fd.command_buffer, object.vk_index_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }
} // namespace application
