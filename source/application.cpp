#include "application.hpp"
#include "math.hpp"

#include <array>
#include <imgui.h>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <map>
#include <ranges>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace application {
    enum class Figure {
        Cube,
        Cone,
        Cone2,
        Floor,
        Ceil,
        LeftWall,
        RightWall,
        FrontWall,
        BackWall,
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
        float shininess; // offset 64
        float _padding0[3]; // offset 68..80 (vec3 в std140 выравнивается по 16)
        float specularColor[3]; // offset 80
        float _padding1; // offset 92  -> размер 96
    };

    static_assert(sizeof(ModelUniforms) == 96);
    static_assert(offsetof(ModelUniforms, shininess) == 64);
    static_assert(offsetof(ModelUniforms, specularColor) == 80);

    struct LightVizPushConstants {
        math::Mat4 mvp; // offset 0
        float color[4]; // offset 64
    };

    static_assert(sizeof(LightVizPushConstants) == 80);

    struct PipelineConsts {
        uint32_t flags; // offset 0
        float time; // offset 4
        float _pad[2]; // offset 8..15 -> размер 16
    };

    static_assert(sizeof(PipelineConsts) == 16);

    constexpr uint32_t FLAG_FANCY = 1u;

    struct Vertex {
        float pos[3];
        float normal[3];
        float uv[2];
    };

    namespace {
        struct UploadContext {
            static constexpr VkDeviceSize staging_buffer_size = 64ull * 1024ull * 1024ull;

            VkDevice device = VK_NULL_HANDLE;
            VmaAllocator allocator = VK_NULL_HANDLE;
            VkCommandPool command_pool = VK_NULL_HANDLE;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_buffer_allocation = VK_NULL_HANDLE;
            void *mapped_memory = nullptr;
            VkDeviceSize offset = 0;
            bool recording = false;

            bool create(const graphics::internal::Context &context) {
                device = context.device;
                allocator = context.allocator;

                const VkCommandPoolCreateInfo command_pool_info = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    .queueFamilyIndex = context.graphics_queue_index,
                };
                if (vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create upload command pool\n";
                    destroy();
                    return false;
                }

                const VkCommandBufferAllocateInfo command_buffer_info = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .commandPool = command_pool,
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = 1,
                };
                if (vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate upload command buffer\n";
                    destroy();
                    return false;
                }

                const VkFenceCreateInfo fence_info = {
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                };
                if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
                    std::cerr << "Failed to create upload fence\n";
                    destroy();
                    return false;
                }

                const VkBufferCreateInfo staging_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = staging_buffer_size,
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                constexpr VmaAllocationCreateInfo staging_allocation_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo allocation_info{};
                if (vmaCreateBuffer(allocator, &staging_info, &staging_allocation_info,
                                    &staging_buffer, &staging_buffer_allocation,
                                    &allocation_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create staging buffer\n";
                    destroy();
                    return false;
                }
                mapped_memory = allocation_info.pMappedData;
                return true;
            }

            bool begin() {
                if (command_buffer == VK_NULL_HANDLE || recording) {
                    std::cerr << "Upload context is not ready to begin recording\n";
                    return false;
                }

                const VkCommandBufferBeginInfo begin_info = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                };
                if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                    std::cerr << "Failed to begin upload command buffer\n";
                    return false;
                }
                recording = true;
                return true;
            }

            VkDeviceSize stage(const void *data, const size_t size) {
                if (!recording || (data == nullptr && size != 0)) {
                    std::cerr << "Invalid data or upload context is not recording\n";
                    return VK_WHOLE_SIZE;
                }

                if (offset > VK_WHOLE_SIZE - 15) {
                    std::cerr << "Staging buffer offset overflow\n";
                    return VK_WHOLE_SIZE;
                }
                const VkDeviceSize aligned_offset = (offset + 15) & ~VkDeviceSize(15);
                const VkDeviceSize upload_size = static_cast<VkDeviceSize>(size);
                if (aligned_offset > staging_buffer_size ||
                    upload_size > staging_buffer_size - aligned_offset) {
                    std::cerr << "Staging buffer capacity exceeded (64 MiB)\n";
                    return VK_WHOLE_SIZE;
                }

                std::memcpy(static_cast<uint8_t *>(mapped_memory) + aligned_offset, data, size);
                offset = aligned_offset + upload_size;
                return aligned_offset;
            }

            bool end_and_wait() {
                if (!recording) {
                    std::cerr << "Upload context is not recording\n";
                    return false;
                }

                if (offset > 0) {
                    if (vmaFlushAllocation(allocator, staging_buffer_allocation, 0, offset) != VK_SUCCESS) {
                        std::cerr << "Failed to flush staging buffer memory\n";
                        recording = false;
                        return false;
                    }

                    const VkMemoryBarrier memory_barrier = {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
                    };
                    vkCmdPipelineBarrier(command_buffer,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                         0, 1, &memory_barrier, 0, nullptr, 0, nullptr);
                }

                if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                    std::cerr << "Failed to end upload command buffer\n";
                    recording = false;
                    return false;
                }
                recording = false;

                const VkSubmitInfo submit_info = {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &command_buffer,
                };
                if (vkQueueSubmit(graphics::internal::context.graphics_queue, 1,
                                  &submit_info, fence) != VK_SUCCESS) {
                    std::cerr << "Failed to submit upload command buffer\n";
                    return false;
                }
                if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
                    std::cerr << "Failed to wait for upload fence\n";
                    return false;
                }

                if (vkResetCommandPool(device, command_pool, 0) != VK_SUCCESS) {
                    std::cerr << "Failed to reset upload command pool\n";
                    return false;
                }
                if (vkResetFences(device, 1, &fence) != VK_SUCCESS) {
                    std::cerr << "Failed to reset upload fence\n";
                    return false;
                }
                offset = 0;
                return true;
            }

            void destroy() {
                if (staging_buffer != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(allocator, staging_buffer, staging_buffer_allocation);
                    staging_buffer = VK_NULL_HANDLE;
                    staging_buffer_allocation = VK_NULL_HANDLE;
                    mapped_memory = nullptr;
                }
                if (fence != VK_NULL_HANDLE) {
                    vkDestroyFence(device, fence, nullptr);
                    fence = VK_NULL_HANDLE;
                }
                if (command_pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device, command_pool, nullptr);
                    command_pool = VK_NULL_HANDLE;
                    command_buffer = VK_NULL_HANDLE;
                }
                device = VK_NULL_HANDLE;
                allocator = VK_NULL_HANDLE;
                offset = 0;
                recording = false;
            }
        };

        UploadContext upload_context{};

        struct Texture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            uint32_t width = 0;
            uint32_t height = 0;
        };

        struct Material {
            std::string name;
            Texture albedo;
            Texture specular; // опционально: view == VK_NULL_HANDLE -> используется black_texture
            Texture emissive; // опционально: view == VK_NULL_HANDLE -> используется black_texture
            VkDescriptorSet set = VK_NULL_HANDLE;
        };

        Texture black_texture{}; // служебная заглушка 1x1 (0,0,0,255)
        VkSampler albedo_sampler = VK_NULL_HANDLE;

        // pixels: RGBA8, width * height * 4 байт; данные копируются в staging сразу.
        bool create_texture_from_pixels(UploadContext &upload, const stbi_uc *pixels,
                                        const int width, const int height,
                                        const VkFormat format, Texture &texture) {
            if (pixels == nullptr || width <= 0 || height <= 0)
                return false;

            texture.width = static_cast<uint32_t>(width);
            texture.height = static_cast<uint32_t>(height);
            const VkDeviceSize pixel_size = static_cast<VkDeviceSize>(texture.width) *
                                            static_cast<VkDeviceSize>(texture.height) * 4;

            const VkImageCreateInfo image_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = format,
                .extent = {texture.width, texture.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
            constexpr VmaAllocationCreateInfo allocation_info = {
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            if (vmaCreateImage(graphics::internal::context.allocator, &image_info, &allocation_info,
                               &texture.image, &texture.allocation, nullptr) != VK_SUCCESS) {
                std::cerr << "Failed to create texture image\n";
                return false;
            }

            const VkImageViewCreateInfo view_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = texture.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(graphics::internal::context.device, &view_info, nullptr,
                                  &texture.view) != VK_SUCCESS) {
                std::cerr << "Failed to create texture image view\n";
                return false;
            }

            const VkDeviceSize staging_offset = upload.stage(pixels, static_cast<size_t>(pixel_size));
            if (staging_offset == VK_WHOLE_SIZE)
                return false;

            const VkImageMemoryBarrier to_transfer = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture.image,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            vkCmdPipelineBarrier(upload.command_buffer,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_transfer);

            const VkBufferImageCopy copy_region = {
                .bufferOffset = staging_offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {texture.width, texture.height, 1},
            };
            vkCmdCopyBufferToImage(upload.command_buffer, upload.staging_buffer, texture.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

            const VkImageMemoryBarrier to_shader_read = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture.image,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            vkCmdPipelineBarrier(upload.command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_shader_read);
            return true;
        }

        bool create_texture_from_file(UploadContext &upload, const std::string &path,
                                      const VkFormat format, Texture &texture) {
            int width = 0;
            int height = 0;
            int source_channels = 0;
            stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &source_channels, STBI_rgb_alpha);
            if (pixels == nullptr || width <= 0 || height <= 0) {
                std::cerr << "Failed to load texture '" << path << "': " << stbi_failure_reason() << "\n";
                if (pixels != nullptr)
                    stbi_image_free(pixels);
                return false;
            }

            const bool ok = create_texture_from_pixels(upload, pixels, width, height, format, texture);
            stbi_image_free(pixels);
            return ok;
        }

        void destroy_texture(Texture &texture) {
            if (texture.view != VK_NULL_HANDLE) {
                vkDestroyImageView(graphics::internal::context.device, texture.view, nullptr);
                texture.view = VK_NULL_HANDLE;
            }
            if (texture.image != VK_NULL_HANDLE) {
                vmaDestroyImage(graphics::internal::context.allocator, texture.image, texture.allocation);
                texture.image = VK_NULL_HANDLE;
                texture.allocation = VK_NULL_HANDLE;
            }
            texture.width = 0;
            texture.height = 0;
        }
    }

    struct Object {
        uint32_t vk_index_count{};

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        uint32_t material_index = 3;

        Object() = default;

        bool create_vertex_and_index_buffers(
            const graphics::internal::Context &context,
            const std::vector<Vertex> &vertexes,
            const std::vector<uint32_t> &indexes
        ) {
            const VkDeviceSize vertex_size = sizeof(Vertex) * vertexes.size();
            const VkBufferCreateInfo vertex_buffer = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = vertex_size,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            constexpr VmaAllocationCreateInfo vertex_buffer_allocation = {
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

            const VkDeviceSize vertex_offset = upload_context.stage(vertexes.data(),
                                                                    static_cast<size_t>(vertex_size));
            if (vertex_offset == VK_WHOLE_SIZE) {
                vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
                vk_vertex_buffer = VK_NULL_HANDLE;
                vk_vertex_buffer_allocation = VK_NULL_HANDLE;
                return false;
            }
            const VkBufferCopy vertex_copy = {
                .srcOffset = vertex_offset,
                .dstOffset = 0,
                .size = vertex_size,
            };
            vkCmdCopyBuffer(upload_context.command_buffer, upload_context.staging_buffer,
                            vk_vertex_buffer, 1, &vertex_copy);
            std::cout << "Successfully created GPU vertex buffer\n";

            vk_index_count = static_cast<uint32_t>(indexes.size());
            const VkDeviceSize index_size = sizeof(uint32_t) * indexes.size();
            const VkBufferCreateInfo index_buffer = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = index_size,
                .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            constexpr VmaAllocationCreateInfo index_buffer_allocation = {
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            if (vmaCreateBuffer(
                    context.allocator,
                    &index_buffer,
                    &index_buffer_allocation,
                    &vk_index_buffer,
                    &vk_index_buffer_allocation,
                    nullptr
                ) != VK_SUCCESS) {
                std::cerr << "Failed to create and allocate index buffer\n";
                vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
                vk_vertex_buffer = VK_NULL_HANDLE;
                vk_vertex_buffer_allocation = VK_NULL_HANDLE;
                return false;
            }

            const VkDeviceSize index_offset = upload_context.stage(indexes.data(),
                                                                   static_cast<size_t>(index_size));
            if (index_offset == VK_WHOLE_SIZE) {
                vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
                vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
                vk_index_buffer = VK_NULL_HANDLE;
                vk_index_buffer_allocation = VK_NULL_HANDLE;
                vk_vertex_buffer = VK_NULL_HANDLE;
                vk_vertex_buffer_allocation = VK_NULL_HANDLE;
                return false;
            }
            const VkBufferCopy index_copy = {
                .srcOffset = index_offset,
                .dstOffset = 0,
                .size = index_size,
            };
            vkCmdCopyBuffer(upload_context.command_buffer, upload_context.staging_buffer,
                            vk_index_buffer, 1, &index_copy);
            std::cout << "Successfully created GPU index buffer\n";
            return true;
        }
    };

    std::vector<Vertex> make_cone_vertexes(const int n) {
        std::vector<Vertex> array;
        array.reserve(3 * n + 3);

        constexpr float apex_y = 1;
        constexpr float base_y = -1;
        constexpr float radius = 0.6f;
        const float height = apex_y - base_y;
        const float step = 2.0f * std::numbers::pi_v<float> / static_cast<float>(n);

        // 0..n: кольцо основания боковой поверхности, включая дубликат для UV-шва.
        for (int i = 0; i <= n; ++i) {
            // Направление обхода кольца противоположно горизонтальному
            // направлению текстуры, поэтому инвертируем U.
            const float u = 1.0f - static_cast<float>(i) / static_cast<float>(n);
            const float corner = static_cast<float>(i) * step;
            const float x = std::cos(corner) * radius;
            const float z = std::sin(corner) * radius;

            const float nx = height * std::cos(corner);
            const float ny = radius;
            const float nz = height * std::sin(corner);
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);

            array.push_back(Vertex{
                {x, base_y, z},
                {nx / len, ny / len, nz / len},
                {u, 1.0f}
            });
        }

        // n+1..2n: отдельная вершина конуса для каждого сегмента.
        for (int i = 0; i < n; ++i) {
            const float middle_angle = (static_cast<float>(i) + 0.5f) * step;
            const float nx = height * std::cos(middle_angle);
            const float ny = radius;
            const float nz = height * std::sin(middle_angle);
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);

            array.push_back(Vertex{
                {0.0f, apex_y, 0.0f},
                {nx / len, ny / len, nz / len},
                {1.0f - (static_cast<float>(i) + 0.5f) / static_cast<float>(n), 0.0f}
            });
        }

        // 2n+1: центр плоского основания.
        array.push_back(Vertex{
            {0.0f, base_y, 0.0f},
            {0.0f, -1.0f, 0.0f},
            {0.5f, 0.5f}
        });

        // 2n+2..3n+2: отдельное кольцо с нормалями основания и UV-кругом.
        for (int i = 0; i <= n; ++i) {
            const float corner = static_cast<float>(i) * step;
            const float x = std::cos(corner) * radius;
            const float z = std::sin(corner) * radius;

            array.push_back(Vertex{
                {x, base_y, z},
                {0.0f, -1.0f, 0.0f},
                {0.5f + 0.5f * std::cos(corner), 0.5f + 0.5f * std::sin(corner)}
            });
        }

        return array;
    }

    std::vector<uint32_t> make_cone_indexes(int n) {
        std::vector<uint32_t> indexes;
        indexes.reserve(n * 6);

        const uint32_t side_ring = 0;
        const uint32_t side_apex = static_cast<uint32_t>(n + 1);
        const uint32_t base_center = static_cast<uint32_t>(2 * n + 1);
        const uint32_t base_ring = static_cast<uint32_t>(2 * n + 2);

        for (int i = 0; i < n; ++i) {
            const uint32_t current = side_ring + static_cast<uint32_t>(i);
            const uint32_t next = current + 1;
            const uint32_t apex = side_apex + static_cast<uint32_t>(i);
            // Сохраняем исходный порядок обхода бокового треугольника.
            indexes.insert(indexes.end(), {apex, current, next});

            const uint32_t base_current = base_ring + static_cast<uint32_t>(i);
            const uint32_t base_next = base_current + 1;
            // Сохраняем исходный порядок обхода треугольника основания.
            indexes.insert(indexes.end(), {base_center, base_current, base_next});
        }

        return indexes;
    }

    std::vector<Vertex> make_cube_vertexes() {
        constexpr float s = 0.3f;
        constexpr std::array<std::array<float, 2>, 4> face_uv = {
            {
                {1.0f, 0.0f}, // TR
                {0.0f, 0.0f}, // TL
                {0.0f, 1.0f}, // BL
                {1.0f, 1.0f}, // BR
            }
        };

        std::vector<Vertex> array;
        array.reserve(24);

        auto push_face = [&](const std::array<std::array<float, 3>, 4> &positions,
                             const std::array<float, 3> &normal) {
            for (size_t i = 0; i < positions.size(); ++i) {
                array.push_back(Vertex{
                    {positions[i][0], positions[i][1], positions[i][2]},
                    {normal[0], normal[1], normal[2]},
                    {face_uv[i][0], face_uv[i][1]}
                });
            }
        };

        // Каждая грань перечислена снаружи в порядке TR, TL, BL, BR.
        // FRONT (+Z)
        push_face({{{s, s, s}, {-s, s, s}, {-s, -s, s}, {s, -s, s}}}, {0, 0, 1});
        // BACK (-Z)
        push_face({{{-s, s, -s}, {s, s, -s}, {s, -s, -s}, {-s, -s, -s}}}, {0, 0, -1});
        // RIGHT (+X)
        push_face({{{s, s, -s}, {s, s, s}, {s, -s, s}, {s, -s, -s}}}, {1, 0, 0});
        // LEFT (-X)
        push_face({{{-s, s, s}, {-s, s, -s}, {-s, -s, -s}, {-s, -s, s}}}, {-1, 0, 0});
        // TOP (+Y)
        push_face({{{s, s, -s}, {-s, s, -s}, {-s, s, s}, {s, s, s}}}, {0, 1, 0});
        // BOTTOM (-Y)
        push_face({{{s, -s, s}, {-s, -s, s}, {-s, -s, -s}, {s, -s, -s}}}, {0, -1, 0});

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

    constexpr float HALF_SIZE_ROOM_SURFACE = 10.0f; // квадрат 20x20
    constexpr float FLOOR_Y = -2.5f;
    constexpr float CEIL_Y = 5.0f;
    constexpr float FLOOR_UV_REPEAT = 4.0f; // UV 0..4, sampler с REPEAT

    std::vector<Vertex> make_floor_vertexes() {
        constexpr float h = HALF_SIZE_ROOM_SURFACE;
        constexpr float y = FLOOR_Y;
        constexpr float uv = FLOOR_UV_REPEAT;

        return {
            Vertex{{h, y, -h}, {0.0f, 1.0f, 0.0f}, {uv, 0.0f}}, // TR
            Vertex{{-h, y, -h}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}}, // TL
            Vertex{{-h, y, h}, {0.0f, 1.0f, 0.0f}, {0.0f, uv}}, // BL
            Vertex{{h, y, h}, {0.0f, 1.0f, 0.0f}, {uv, uv}}, // BR
        };
    }

    std::vector<uint32_t> make_floor_indexes() {
        return {0, 2, 1, 0, 3, 2};
    }


    std::vector<Vertex> make_wall_vertexes() {
        constexpr float h = HALF_SIZE_ROOM_SURFACE;
        constexpr float floor_y = FLOOR_Y;
        constexpr float ceil_y = CEIL_Y;
        constexpr float uv = FLOOR_UV_REPEAT;

        return {
            Vertex{{-h, floor_y, -h}, {1, 0, 0}, {uv, 0}},
            Vertex{{-h, floor_y, h}, {1, 0, 0}, {0, 0}},
            Vertex{{-h, ceil_y, h}, {1, 0, 0}, {0, uv}},
            Vertex{{-h, ceil_y, -h}, {1, 0, 0}, {uv, uv}},
        };
    }

    std::vector<uint32_t> make_wall_indexes() {
        return {0, 1, 2, 0, 2, 3};
    }

    std::vector<Vertex> make_line_cone_vertexes(const int segments) {
        std::vector<Vertex> arr;
        arr.reserve(segments * 4);

        for (int i = 0; i < segments; ++i) {
            const float a0 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i)
                             / static_cast<float>(segments);
            const float a1 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i + 1)
                             / static_cast<float>(segments);

            const float x0 = std::cos(a0), y0 = std::sin(a0);
            const float x1 = std::cos(a1), y1 = std::sin(a1);

            // линия apex -> точка на окружности
            arr.push_back(Vertex{{0.0f, 0.0f, 0.0f}, {0, 0, 0}, {0, 0}});
            arr.push_back(Vertex{{x0, y0, 1.0f}, {0, 0, 0}, {0, 0}});

            // линия по окружности
            arr.push_back(Vertex{{x0, y0, 1.0f}, {0, 0, 0}, {0, 0}});
            arr.push_back(Vertex{{x1, y1, 1.0f}, {0, 0, 0}, {0, 0}});
        }

        return arr;
    }

    math::Mat4 rotation_align_z_with(const math::Vec3 &dir) {
        const math::Vec3 z = math::Vec3::normalize(dir);
        const math::Vec3 up = (std::abs(z.y) < 0.99f)
                                  ? math::Vec3{0, 1, 0}
                                  : math::Vec3{1, 0, 0};
        const math::Vec3 x = math::Vec3::normalize(math::Vec3::cross(up, z));
        const math::Vec3 y = math::Vec3::cross(z, x);

        math::Mat4 m{};
        // column 0 = x, column 1 = y, column 2 = z
        m.matrix[0] = x.x;
        m.matrix[1] = x.y;
        m.matrix[2] = x.z;
        m.matrix[4] = y.x;
        m.matrix[5] = y.y;
        m.matrix[6] = y.z;
        m.matrix[8] = z.x;
        m.matrix[9] = z.y;
        m.matrix[10] = z.z;
        m.matrix[12] = 0;
        m.matrix[13] = 0;
        m.matrix[14] = 0;
        m.matrix[15] = 1;
        return m;
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
        constexpr uint32_t MAX_MODELS = 1024;

        static_assert(static_cast<uint32_t>(Figure::Floor) < MAX_MODELS);

        std::map<Figure, Object> objects{};

        VkBuffer vk_model_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
        uint8_t *vk_model_uniform_memory = nullptr;
        VkDeviceSize uniforms_stride = 0;

        struct MaterialSource {
            const char *name;
            const char *albedo_path;
            const char *specular_path; // nullptr -> заглушка
            const char *emissive_path; // nullptr -> заглушка
        };

        constexpr MaterialSource material_sources[] = {
            {"UV test", "../assets/test_albedo.png", nullptr, nullptr},
            {
                "Bricks", "../assets/mat_bricks.png",
                "../assets/mat_bricks_spec.png", "../assets/mat_bricks_emissive.png"
            },
            {"Stripes", "../assets/mat_stripes.png", "../assets/mat_stripes_spec.png", nullptr},
            {
                "Rings", "../assets/mat_rings.png",
                "../assets/mat_rings_spec.png", "../assets/mat_rings_emissive.png"
            },
            {"Floor", "../assets/mat_floor.png", "../assets/mat_floor_spec.png", nullptr},
        };
        constexpr size_t material_source_count = sizeof(material_sources) / sizeof(material_sources[0]);
        constexpr uint32_t MAX_MATERIALS = 16;
        static_assert(material_source_count <= MAX_MATERIALS);

        std::vector<Material> materials{};

        uint32_t model_index(const Figure figure) {
            return static_cast<uint32_t>(figure);
        }

        ModelUniforms *model_uniforms_at(const Figure figure) {
            return reinterpret_cast<ModelUniforms *>(
                vk_model_uniform_memory + model_index(figure) * uniforms_stride
            );
        }

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

        VkPipelineLayout vk_cone_layout = VK_NULL_HANDLE;
        VkPipeline vk_cone_pipeline = VK_NULL_HANDLE;

        VkShaderModule vk_light_viz_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_light_viz_fragment_shader = VK_NULL_HANDLE;

        VkPipelineLayout vk_light_viz_layout = VK_NULL_HANDLE;
        VkPipeline vk_light_cube_pipeline = VK_NULL_HANDLE; // TRIANGLE_LIST
        VkPipeline vk_light_line_pipeline = VK_NULL_HANDLE; // LINE_LIST

        VkBuffer vk_line_cone_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_line_cone_vertex_buffer_allocation = VK_NULL_HANDLE;
        uint32_t vk_line_cone_vertex_count = 0;

        bool ui_show_point_lights = true;
        bool ui_show_spot_lights = true;
        bool ui_show_spot_cones = true;

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
        float ui_dir_intensity = 1;

        float ui_specular_color[3] = {1.0f, 0.95f, 0.8f};
        float ui_shininess = 64.0f;

        bool ui_fancy_enabled = false;
        Figure ui_fancy_figure = Figure::Cone2;
        float current_time = 0.0f;
    }

    bool initialize() {
        auto &context = graphics::internal::context;

        if (!upload_context.create(context))
            return false;

        struct UploadContextGuard {
            UploadContext &upload;
            bool keep = false;

            ~UploadContextGuard() {
                if (!keep)
                    upload.destroy();
            }
        } upload_guard{upload_context};

        if (!upload_context.begin())
            return false;

        {
            constexpr stbi_uc black_pixel[4] = {0, 0, 0, 255};
            if (!create_texture_from_pixels(upload_context, black_pixel, 1, 1,
                                            VK_FORMAT_R8G8B8A8_UNORM, black_texture))
                return false;
        }

        materials.resize(material_source_count);
        for (size_t i = 0; i < material_source_count; ++i) {
            const MaterialSource &source = material_sources[i];
            Material &material = materials[i];
            material.name = source.name;
            if (!create_texture_from_file(upload_context, source.albedo_path,
                                          VK_FORMAT_R8G8B8A8_SRGB, material.albedo))
                return false;
            if (source.specular_path != nullptr &&
                !create_texture_from_file(upload_context, source.specular_path,
                                          VK_FORMAT_R8G8B8A8_UNORM, material.specular))
                return false;
            if (source.emissive_path != nullptr &&
                !create_texture_from_file(upload_context, source.emissive_path,
                                          VK_FORMAT_R8G8B8A8_SRGB, material.emissive))
                return false;
        }

        const VkSamplerCreateInfo sampler_info = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        if (vkCreateSampler(context.device, &sampler_info, nullptr, &albedo_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create albedo sampler\n";
            return false;
        }

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

        auto cone2 = Object();
        if (!cone2.create_vertex_and_index_buffers(context, cone_vertexes, cone_indexes))
            return false;

        const auto floor_vertexes = make_floor_vertexes();
        const auto floor_indexes = make_floor_indexes();

        auto floor = Object();
        if (!floor.create_vertex_and_index_buffers(context, floor_vertexes, floor_indexes))
            return false;

        auto ceil = Object();
        if (!ceil.create_vertex_and_index_buffers(context, floor_vertexes, floor_indexes))
            return false;

        for (uint32_t i = 0; i < materials.size(); ++i) {
            if (materials[i].name == "Floor") {
                floor.material_index = i;
                ceil.material_index = i;
            }
        }

        const auto wall_vertexes = make_wall_vertexes();
        const auto wall_indexes = make_wall_indexes();

        auto left_wall = Object();
        if (!left_wall.create_vertex_and_index_buffers(context, wall_vertexes, wall_indexes))
            return false;

        auto right_wall = Object();
        if (!right_wall.create_vertex_and_index_buffers(context, wall_vertexes, wall_indexes))
            return false;

        auto front_wall = Object();
        if (!front_wall.create_vertex_and_index_buffers(context, wall_vertexes, wall_indexes))
            return false;

        auto back_wall = Object();
        if (!back_wall.create_vertex_and_index_buffers(context, wall_vertexes, wall_indexes))
            return false;

        for (uint32_t i = 0; i < materials.size(); ++i) {
            if (materials[i].name == "Bricks") {
                left_wall.material_index = i;
                right_wall.material_index = i;
                front_wall.material_index = i;
                back_wall.material_index = i;
            }
        }

        objects[Figure::Cone] = cone;
        objects[Figure::Cube] = cube;
        objects[Figure::Cone2] = cone2;
        objects[Figure::Floor] = floor;
        objects[Figure::Ceil] = ceil;
        objects[Figure::LeftWall] = left_wall;
        objects[Figure::RightWall] = right_wall;
        objects[Figure::FrontWall] = front_wall;
        objects[Figure::BackWall] = back_wall;

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
                .offset = offsetof(Vertex, normal),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
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

        const VkDescriptorSetLayoutBinding model_bindings[4] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        };
        const VkDescriptorSetLayoutCreateInfo model_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = model_bindings,
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
        const VkPushConstantRange cone_push_range = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PipelineConsts),
        };
        const VkPipelineLayoutCreateInfo layout_create_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 2,
            .pSetLayouts = set_layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &cone_push_range,
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
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 4},
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 4 + MAX_MATERIALS},
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4},
            {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 4 + 3 * MAX_MATERIALS},
        };
        const VkDescriptorPoolCreateInfo descriptor_pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 4 + MAX_MATERIALS,
            .poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
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

        {
            VkPhysicalDeviceProperties device_properties{};
            vkGetPhysicalDeviceProperties(context.physical_device, &device_properties);

            const VkDeviceSize alignment = device_properties.limits.minUniformBufferOffsetAlignment;
            uniforms_stride = (sizeof(ModelUniforms) + alignment - 1) & ~(alignment - 1);

            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = uniforms_stride * MAX_MODELS,
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
                    &vk_model_uniform_buffer, &vk_model_uniform_buffer_allocation,
                    &out_info) != VK_SUCCESS) {
                std::cerr << "Failed to create model uniform buffer\n";
                return false;
            }
            vk_model_uniform_memory = static_cast<uint8_t *>(out_info.pMappedData);
            std::memset(vk_model_uniform_memory, 0, static_cast<size_t>(uniforms_stride * MAX_MODELS));

            const std::vector<VkDescriptorSetLayout> material_layouts(
                materials.size(), vk_model_descriptor_set_layout);
            std::vector<VkDescriptorSet> material_sets(materials.size(), VK_NULL_HANDLE);
            const VkDescriptorSetAllocateInfo descriptor_set_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = static_cast<uint32_t>(material_layouts.size()),
                .pSetLayouts = material_layouts.data(),
            };
            if (vkAllocateDescriptorSets(context.device, &descriptor_set_info, material_sets.data()) !=
                VK_SUCCESS) {
                std::cerr << "Failed to allocate material descriptor sets\n";
                return false;
            }

            const VkDescriptorBufferInfo descriptor_buffer_info = {
                .buffer = vk_model_uniform_buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            };
            // Для каждого материала: albedo (binding 1), specular (2), emissive (3).
            // Отсутствующие specular/emissive заменяются чёрной заглушкой.
            std::vector<VkDescriptorImageInfo> image_descriptor_infos(materials.size() * 3);
            std::vector<VkWriteDescriptorSet> descriptor_writes;
            descriptor_writes.reserve(materials.size() * 4);
            for (size_t i = 0; i < materials.size(); ++i) {
                materials[i].set = material_sets[i];

                const VkImageView views[3] = {
                    materials[i].albedo.view,
                    materials[i].specular.view != VK_NULL_HANDLE ? materials[i].specular.view : black_texture.view,
                    materials[i].emissive.view != VK_NULL_HANDLE ? materials[i].emissive.view : black_texture.view,
                };

                descriptor_writes.push_back({
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = materials[i].set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &descriptor_buffer_info,
                });
                for (uint32_t k = 0; k < 3; ++k) {
                    image_descriptor_infos[i * 3 + k] = {
                        .sampler = albedo_sampler,
                        .imageView = views[k],
                        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    };
                    descriptor_writes.push_back({
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = materials[i].set,
                        .dstBinding = 1 + k,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &image_descriptor_infos[i * 3 + k],
                    });
                }
            }
            vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(descriptor_writes.size()),
                                   descriptor_writes.data(), 0, nullptr);
            std::cout << "Successfully created dynamic model uniform buffer and material descriptor sets\n";
        }


        vk_light_viz_vertex_shader = load_shader_module("../shaders/light_viz.vert.spv");
        vk_light_viz_fragment_shader = load_shader_module("../shaders/light_viz.frag.spv");
        if (vk_light_viz_vertex_shader == VK_NULL_HANDLE || vk_light_viz_fragment_shader == VK_NULL_HANDLE) {
            std::cerr << "Failed to load light viz shaders\n";
            return false;
        }

        // --- Line cone vertex buffer ---
        {
            const auto line_verts = make_line_cone_vertexes(24);
            vk_line_cone_vertex_count = static_cast<uint32_t>(line_verts.size());

            const VkDeviceSize line_vertex_size = sizeof(Vertex) * line_verts.size();
            const VkBufferCreateInfo bi = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = line_vertex_size,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            constexpr VmaAllocationCreateInfo ai = {
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            if (vmaCreateBuffer(context.allocator, &bi, &ai,
                                &vk_line_cone_vertex_buffer,
                                &vk_line_cone_vertex_buffer_allocation,
                                nullptr) != VK_SUCCESS) {
                std::cerr << "Failed to create line cone vertex buffer\n";
                return false;
            }

            const VkDeviceSize line_vertex_offset = upload_context.stage(
                line_verts.data(), static_cast<size_t>(line_vertex_size));
            if (line_vertex_offset == VK_WHOLE_SIZE) {
                vmaDestroyBuffer(context.allocator, vk_line_cone_vertex_buffer,
                                 vk_line_cone_vertex_buffer_allocation);
                vk_line_cone_vertex_buffer = VK_NULL_HANDLE;
                vk_line_cone_vertex_buffer_allocation = VK_NULL_HANDLE;
                return false;
            }
            const VkBufferCopy line_vertex_copy = {
                .srcOffset = line_vertex_offset,
                .dstOffset = 0,
                .size = line_vertex_size,
            };
            vkCmdCopyBuffer(upload_context.command_buffer, upload_context.staging_buffer,
                            vk_line_cone_vertex_buffer, 1, &line_vertex_copy);
        }

        if (!upload_context.end_and_wait())
            return false;

        // --- Pipeline layout с push-constant ---
        {
            const VkPushConstantRange push_range = {
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(LightVizPushConstants),
            };
            const VkPipelineLayoutCreateInfo pli = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 0,
                .pSetLayouts = nullptr,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_range,
            };
            if (vkCreatePipelineLayout(context.device, &pli, nullptr, &vk_light_viz_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create light viz pipeline layout\n";
                return false;
            }
        }

        // --- Общий vertex input state ---
        const VkVertexInputBindingDescription viz_bindings[1] = {
            {.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
        };
        const VkVertexInputAttributeDescription viz_attrs[1] = {
            {
                .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, pos)
            },
        };
        const VkPipelineVertexInputStateCreateInfo viz_vertex_input = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = viz_bindings,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = viz_attrs,
        };

        const VkPipelineShaderStageCreateInfo viz_stages[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vk_light_viz_vertex_shader, .pName = "main"
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = vk_light_viz_fragment_shader, .pName = "main"
            },
        };

        const VkPipelineViewportStateCreateInfo viz_viewport = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .scissorCount = 1,
        };
        const VkPipelineRasterizationStateCreateInfo viz_raster = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE, // кубик рисуем без culling
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        const VkPipelineMultisampleStateCreateInfo viz_msaa = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        const VkPipelineColorBlendAttachmentState viz_blend_att = {
            .colorWriteMask = 0xF,
        };
        const VkPipelineColorBlendStateCreateInfo viz_blend = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &viz_blend_att,
        };
        const VkDynamicState viz_dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        const VkPipelineDynamicStateCreateInfo viz_dyn = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2, .pDynamicStates = viz_dyn_states,
        };

        // Solid cube pipeline (TRIANGLE_LIST, depth write on)
        {
            const VkPipelineInputAssemblyStateCreateInfo ia = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };
            const VkPipelineDepthStencilStateCreateInfo ds = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
            };
            const VkGraphicsPipelineCreateInfo gpci = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2, .pStages = viz_stages,
                .pVertexInputState = &viz_vertex_input,
                .pInputAssemblyState = &ia,
                .pViewportState = &viz_viewport,
                .pRasterizationState = &viz_raster,
                .pMultisampleState = &viz_msaa,
                .pDepthStencilState = &ds,
                .pColorBlendState = &viz_blend,
                .pDynamicState = &viz_dyn,
                .layout = vk_light_viz_layout,
                .renderPass = context.render_pass, .subpass = 0,
            };
            if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &gpci, nullptr,
                                          &vk_light_cube_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create light cube pipeline\n";
                return false;
            }
        }

        // Line pipeline (LINE_LIST, depth write off — чтобы не «спорить» с кубиком)
        {
            const VkPipelineInputAssemblyStateCreateInfo ia = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            };
            const VkPipelineDepthStencilStateCreateInfo ds = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_FALSE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
            };
            const VkGraphicsPipelineCreateInfo gpci = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2, .pStages = viz_stages,
                .pVertexInputState = &viz_vertex_input,
                .pInputAssemblyState = &ia,
                .pViewportState = &viz_viewport,
                .pRasterizationState = &viz_raster,
                .pMultisampleState = &viz_msaa,
                .pDepthStencilState = &ds,
                .pColorBlendState = &viz_blend,
                .pDynamicState = &viz_dyn,
                .layout = vk_light_viz_layout,
                .renderPass = context.render_pass, .subpass = 0,
            };
            if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &gpci, nullptr,
                                          &vk_light_line_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create light line pipeline\n";
                return false;
            }
        }

        upload_guard.keep = true;
        return true;
    }

    void shutdown() {
        auto &context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (vk_light_cube_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, vk_light_cube_pipeline, nullptr);
            vk_light_cube_pipeline = VK_NULL_HANDLE;
        }
        if (vk_light_line_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, vk_light_line_pipeline, nullptr);
            vk_light_line_pipeline = VK_NULL_HANDLE;
        }
        if (vk_light_viz_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context.device, vk_light_viz_layout, nullptr);
            vk_light_viz_layout = VK_NULL_HANDLE;
        }
        if (vk_light_viz_vertex_shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, vk_light_viz_vertex_shader, nullptr);
            vk_light_viz_vertex_shader = VK_NULL_HANDLE;
        }
        if (vk_light_viz_fragment_shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, vk_light_viz_fragment_shader, nullptr);
            vk_light_viz_fragment_shader = VK_NULL_HANDLE;
        }
        if (vk_line_cone_vertex_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_line_cone_vertex_buffer,
                             vk_line_cone_vertex_buffer_allocation);
            vk_line_cone_vertex_buffer = VK_NULL_HANDLE;
            vk_line_cone_vertex_buffer_allocation = VK_NULL_HANDLE;
        }

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
            for (auto &material: materials)
                material.set = VK_NULL_HANDLE;
        }

        if (albedo_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(context.device, albedo_sampler, nullptr);
            albedo_sampler = VK_NULL_HANDLE;
        }
        for (auto &material: materials) {
            destroy_texture(material.albedo);
            destroy_texture(material.specular);
            destroy_texture(material.emissive);
        }
        materials.clear();
        destroy_texture(black_texture);


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

        if (vk_model_uniform_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_model_uniform_buffer, vk_model_uniform_buffer_allocation);
            vk_model_uniform_buffer = VK_NULL_HANDLE;
            vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
            vk_model_uniform_memory = nullptr;
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

        upload_context.destroy();
    }

    void update(const double time) {
        ImGui::ShowDemoWindow();

        current_time = static_cast<float>(time);

        static double last_time = 0;
        const auto delta_time = time - last_time;
        last_time = time;

        ImGui::Begin("Controls");

        ImGui::Separator();
        ImGui::Text("Light visualization");
        ImGui::Checkbox("Show point lights", &ui_show_point_lights);
        ImGui::Checkbox("Show spot lights", &ui_show_spot_lights);
        ImGui::Checkbox("Show spot cones", &ui_show_spot_cones);

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
        ImGui::Text("Ambient");
        ImGui::SliderFloat("Strength", &ui_ambient_strength, 0, 1);
        ImGui::ColorEdit3("Ambient color", ui_ambient_color);

        ImGui::Separator();
        ImGui::Text("Directional light");
        ImGui::SliderFloat3("Direction", ui_dir_direction, -1, 1);
        ImGui::ColorEdit3("Dir color", ui_dir_color);
        ImGui::SliderFloat("Dir intensity", &ui_dir_intensity, 0, 5);

        ImGui::Separator();
        ImGui::Text("Point lights: %zu", point_lights.size());
        if (ImGui::Button("Add point light")) {
            PointLightUniform pl{};
            pl.position[0] = 0.0f;
            pl.position[1] = 1.5f;
            pl.position[2] = 0.0f;
            pl.color[0] = 1.0f;
            pl.color[1] = 0.6f;
            pl.color[2] = 0.3f;
            pl.intensity = 8.0f;
            pl.radius = 6.0f;
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
                ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 50.0f);
                ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.1f, 20.0f);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Text("Spot lights: %zu", spot_lights.size());
        if (ImGui::Button("Add spot light")) {
            SpotLightUniform sl{};
            sl.position[0] = 0.0f;
            sl.position[1] = 2.0f;
            sl.position[2] = 0.0f;
            sl.direction[0] = 0.0f;
            sl.direction[1] = -1.0f;
            sl.direction[2] = 0.0f;
            sl.color[0] = 1.0f;
            sl.color[1] = 1.0f;
            sl.color[2] = 1.0f;
            sl.intensity = 30.0f;
            sl.innerAngle = 0.25f; // 14 градусов
            sl.outerAngle = 0.45f; // 26 градусов
            sl.radius = 10.0f;
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
                ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 100.0f);
                ImGui::SliderFloat("Inner angle", &spot_lights[i].innerAngle, 0.0f, 1.55f);
                ImGui::SliderFloat("Outer angle", &spot_lights[i].outerAngle, 0.0f, 1.55f);
                ImGui::SliderFloat("Radius", &spot_lights[i].radius, 0.0f, 20.0f);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Text("Material");
        ImGui::ColorEdit3("Specular color", ui_specular_color);
        ImGui::SliderFloat("Shininess", &ui_shininess, 1, 256);

        ImGui::Separator();
        ImGui::Text("Textures");
        {
            struct FigureEntry {
                Figure figure;
                const char *name;
            };
            static constexpr FigureEntry figure_entries[] = {
                {Figure::Cube, "Cube"},
                {Figure::Cone, "Cone"},
                {Figure::Cone2, "Cone 2"},
            };
            for (const auto &entry: figure_entries) {
                const auto it = objects.find(entry.figure);
                if (it == objects.end() || materials.empty())
                    continue;

                Object &object = it->second;
                if (object.material_index >= materials.size())
                    object.material_index = 3;

                const std::string combo_label = std::string(entry.name) + "##material_combo";
                if (ImGui::BeginCombo(combo_label.c_str(), materials[object.material_index].name.c_str())) {
                    for (uint32_t i = 0; i < materials.size(); ++i) {
                        const bool selected = (object.material_index == i);
                        const std::string item_label = materials[i].name + "##" + entry.name +
                                                       "_material_" + std::to_string(i);
                        if (ImGui::Selectable(item_label.c_str(), selected))
                            object.material_index = i;
                        if (selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

            ImGui::Checkbox("Fancy sampling", &ui_fancy_enabled);
            const char *fancy_preview = "Cone 2";
            for (const auto &entry: figure_entries) {
                if (entry.figure == ui_fancy_figure)
                    fancy_preview = entry.name;
            }
            if (ImGui::BeginCombo("Fancy object##fancy_object_combo", fancy_preview)) {
                for (const auto &entry: figure_entries) {
                    const bool selected = (entry.figure == ui_fancy_figure);
                    const std::string item_label = std::string(entry.name) + "##fancy_object_" +
                                                   std::to_string(static_cast<int>(entry.figure));
                    if (ImGui::Selectable(item_label.c_str(), selected))
                        ui_fancy_figure = entry.figure;
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

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
        const auto animated_pos_cube = create_animated_position_cube(t, r);
        const math::Vec3 pos_cube = {
            animated_pos_cube.x + 1.5f,
            animated_pos_cube.y,
            animated_pos_cube.z,
        };

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

        const auto animated_pos_cone2 = create_animated_position_cone(
            t + 1.7f,
            r * 0.75f,
            y_amp * 0.5f
        );

        const math::Vec3 pos_cone2 = ui_playing
                                         ? animated_pos_cone2
                                         : math::Vec3{-1.5f, 0.0f, 0.0f};

        const math::Mat4 cone2_model =
                math::Mat4::translate(pos_cone2.x, pos_cone2.y, pos_cone2.z)
                * math::Mat4::rotate_y(ui_rotation[1] * deg2rad + 0.8f)
                * math::Mat4::rotate_x(ui_rotation[0] * deg2rad)
                * math::Mat4::rotate_z(ui_rotation[2] * deg2rad)
                * math::Mat4::scale(ui_scale[0] * 0.7f,
                                    ui_scale[1] * 0.7f,
                                    ui_scale[2] * 0.7f);

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

        const auto ceil_model = math::Mat4::rotate_x(std::numbers::pi_v<float>);

        const auto right_wall_model = math::Mat4::rotate_y(std::numbers::pi_v<float>);

        const auto front_wall_model = math::Mat4::rotate_y(-std::numbers::pi_v<float> / 2);

        const auto back_wall_model = math::Mat4::rotate_y(std::numbers::pi_v<float> / 2);

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
            ModelUniforms *mm = model_uniforms_at(Figure::Cone);
            mm->model = cone_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::Cone2);
            mm->model = cone2_model;
            mm->shininess = ui_shininess * 0.7f;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::Cube);
            mm->model = cube_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::Floor);
            mm->model = math::Mat4::identity();
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::Ceil);
            mm->model = ceil_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::LeftWall);
            mm->model = math::Mat4::identity();
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::RightWall);
            mm->model = right_wall_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::FrontWall);
            mm->model = front_wall_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
        }
        {
            ModelUniforms *mm = model_uniforms_at(Figure::BackWall);
            mm->model = back_wall_model;
            mm->shininess = ui_shininess;
            mm->specularColor[0] = ui_specular_color[0];
            mm->specularColor[1] = ui_specular_color[1];
            mm->specularColor[2] = ui_specular_color[2];
            mm->_padding1 = 0.0f;
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

        for (const auto &[figure, object]: objects) {
            const VkDeviceSize device_size = 0;
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.vk_vertex_buffer, &device_size);
            vkCmdBindIndexBuffer(fd.command_buffer, object.vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

            const uint32_t dynamic_offset = static_cast<uint32_t>(model_index(figure) * uniforms_stride);
            const size_t material_index = object.material_index < materials.size()
                                              ? object.material_index
                                              : 0;
            const VkDescriptorSet descriptor_sets[2] = {
                vk_scene_descriptor_set,
                materials[material_index].set,
            };
            vkCmdBindDescriptorSets(
                fd.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_cone_layout,
                0,
                2,
                descriptor_sets,
                1,
                &dynamic_offset
            );

            PipelineConsts pipeline_consts{};
            pipeline_consts.flags = (ui_fancy_enabled && figure == ui_fancy_figure) ? FLAG_FANCY : 0u;
            pipeline_consts.time = current_time;
            vkCmdPushConstants(fd.command_buffer, vk_cone_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pipeline_consts), &pipeline_consts);

            vkCmdDrawIndexed(fd.command_buffer, object.vk_index_count, 1, 0, 0, 0);
        }
        if (ui_show_point_lights || ui_show_spot_lights || ui_show_spot_cones) {
            const math::Mat4 vp = vk_scene_uniform_memory->proj * vk_scene_uniform_memory->view;

            const VkDeviceSize offset0 = 0;
            constexpr VkShaderStageFlags push_flags =
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            if ((ui_show_point_lights && !point_lights.empty()) ||
                (ui_show_spot_lights && !spot_lights.empty())) {
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  vk_light_cube_pipeline);

                const auto &cube = objects[Figure::Cube];
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &cube.vk_vertex_buffer, &offset0);
                vkCmdBindIndexBuffer(fd.command_buffer, cube.vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

                constexpr float viz_cube_half = 0.15f;
                constexpr float original_cube_half = 0.3f;
                const float viz_scale = viz_cube_half / original_cube_half;

                const math::Mat4 cube_scale = math::Mat4::scale(viz_scale, viz_scale, viz_scale);

                auto draw_cube = [&](float x, float y, float z, const float col[3]) {
                    LightVizPushConstants pc{};
                    pc.mvp = vp * math::Mat4::translate(x, y, z) * cube_scale;
                    pc.color[0] = col[0];
                    pc.color[1] = col[1];
                    pc.color[2] = col[2];
                    pc.color[3] = 1.0f;
                    vkCmdPushConstants(fd.command_buffer, vk_light_viz_layout,
                                       push_flags, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(fd.command_buffer, cube.vk_index_count, 1, 0, 0, 0);
                };

                if (ui_show_point_lights) {
                    for (const auto &pl: point_lights) {
                        draw_cube(pl.position[0], pl.position[1], pl.position[2], pl.color);
                    }
                }
                if (ui_show_spot_lights) {
                    for (const auto &sl: spot_lights) {
                        draw_cube(sl.position[0], sl.position[1], sl.position[2], sl.color);
                    }
                }
            }

            if (ui_show_spot_cones && !spot_lights.empty()) {
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  vk_light_line_pipeline);
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                                       &vk_line_cone_vertex_buffer, &offset0);

                for (const auto &sl: spot_lights) {
                    const math::Vec3 dir = math::Vec3::normalize({
                        sl.direction[0], sl.direction[1], sl.direction[2]
                    });

                    const math::Mat4 rot = rotation_align_z_with(dir);

                    const float len = (sl.radius > 0.0f) ? sl.radius : 2.5f;

                    const float tan_outer = std::tan(sl.outerAngle);
                    const float r_world = tan_outer * len;

                    const math::Mat4 world =
                            math::Mat4::translate(sl.position[0], sl.position[1], sl.position[2])
                            * rot
                            * math::Mat4::scale(r_world, r_world, len);

                    LightVizPushConstants pc{};
                    pc.mvp = vp * world;
                    pc.color[0] = 1.0f;
                    pc.color[1] = 1.0f;
                    pc.color[2] = 1.0f;
                    pc.color[3] = 1.0f;
                    vkCmdPushConstants(fd.command_buffer, vk_light_viz_layout,
                                       push_flags, 0, sizeof(pc), &pc);
                    vkCmdDraw(fd.command_buffer, vk_line_cone_vertex_count, 1, 0, 0);
                }
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }
} // namespace application
