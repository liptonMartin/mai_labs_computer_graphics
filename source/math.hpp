//
// Created by rvova on 03.10.2026.
//

#ifndef VULKAN_STARTER_APP_MATH_HPP
#define VULKAN_STARTER_APP_MATH_HPP
#include <cmath>


namespace math {
    struct Vec3 {
        float x = 0, y = 0, z = 0;

        Vec3() = default;

        Vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {
        }

        static Vec3 normalize(const Vec3 &v) {
            const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (len == 0.0f) return {};
            return {v.x / len, v.y / len, v.z / len};
        }

        static Vec3 cross(const Vec3 &a, const Vec3 &b) {
            return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            };
        }

        static float dot(const Vec3 &a, const Vec3 &b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        Vec3 operator-(const Vec3 &o) const {
            return {x - o.x, y - o.y, z - o.z};
        }
    };

    struct Mat4 {
        // column-major для GLSL
        float matrix[16] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };

        static Mat4 identity() {
            return Mat4{};
        }

        static Mat4 translate(float x, float y, float z) {
            Mat4 result{};
            result.matrix[3 * 4 + 0] = x;
            result.matrix[3 * 4 + 1] = y;
            result.matrix[3 * 4 + 2] = z;
            return result;
        }

        static Mat4 scale(float x, float y, float z) {
            Mat4 result{};
            result.matrix[0 * 4 + 0] = x;
            result.matrix[1 * 4 + 1] = y;
            result.matrix[2 * 4 + 2] = z;
            return result;
        }

        static Mat4 rotate_x(const float a) {
            const auto sin_a = std::sin(a);
            const auto cos_a = std::cos(a);

            Mat4 result{};
            result.matrix[1 * 4 + 1] = cos_a;
            result.matrix[2 * 4 + 2] = cos_a;
            result.matrix[2 * 4 + 1] = -sin_a;
            result.matrix[1 * 4 + 2] = sin_a;

            return result;
        }

        static Mat4 rotate_y(const float a) {
            const auto sin_a = std::sin(a);
            const auto cos_a = std::cos(a);

            Mat4 result{};
            result.matrix[0 * 4 + 0] = cos_a;
            result.matrix[2 * 4 + 2] = cos_a;
            result.matrix[2 * 4 + 0] = sin_a;
            result.matrix[0 * 4 + 2] = -sin_a;

            return result;
        }

        static Mat4 rotate_z(const float a) {
            const auto sin_a = std::sin(a);
            const auto cos_a = std::cos(a);

            Mat4 result{};
            result.matrix[0 * 4 + 0] = cos_a;
            result.matrix[1 * 4 + 1] = cos_a;
            result.matrix[1 * 4 + 0] = -sin_a;
            result.matrix[0 * 4 + 1] = sin_a;

            return result;
        }

        static Mat4 perspective(const float fov_y_rad, const float aspect, const float near, const float far) {
            Mat4 r{};
            const float f = 1 / std::tan(fov_y_rad * 0.5f);
            r.matrix[0 * 4 + 0] = f / aspect;
            r.matrix[1 * 4 + 1] = -f; // flip Y под Vulkan
            r.matrix[2 * 4 + 2] = far / (near - far);
            r.matrix[2 * 4 + 3] = -1;
            r.matrix[3 * 4 + 2] = (far * near) / (near - far);
            r.matrix[3 * 4 + 3] = 0;
            return r;
        }

        static Mat4 ortho(
            const float left,
            const float right,
            const float bottom,
            const float top,
            const float near,
            const float far
        ) {
            Mat4 r{};
            r.matrix[0 * 4 + 0] = 2 / (right - left);
            r.matrix[1 * 4 + 1] = 2 / (top - bottom);
            r.matrix[2 * 4 + 2] = 1 / (near - far);
            r.matrix[3 * 4 + 0] = -(right + left) / (right - left);
            r.matrix[3 * 4 + 1] = -(top + bottom) / (top - bottom);
            r.matrix[3 * 4 + 2] = near / (near - far);
            r.matrix[3 * 4 + 3] = 1;
            return r;
        }

        static Mat4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
            const Vec3 f = Vec3::normalize(center - eye);
            const Vec3 s = Vec3::normalize(Vec3::cross(f, up));
            const Vec3 u = Vec3::cross(s, f);

            Mat4 r{};
            r.matrix[0 * 4 + 0] = s.x;
            r.matrix[1 * 4 + 0] = s.y;
            r.matrix[2 * 4 + 0] = s.z;
            r.matrix[0 * 4 + 1] = u.x;
            r.matrix[1 * 4 + 1] = u.y;
            r.matrix[2 * 4 + 1] = u.z;
            r.matrix[0 * 4 + 2] = -f.x;
            r.matrix[1 * 4 + 2] = -f.y;
            r.matrix[2 * 4 + 2] = -f.z;
            r.matrix[3 * 4 + 0] = -Vec3::dot(s, eye);
            r.matrix[3 * 4 + 1] = -Vec3::dot(u, eye);
            r.matrix[3 * 4 + 2] = Vec3::dot(f, eye);
            r.matrix[3 * 4 + 3] = 1.0f;
            return r;
        }

        Mat4 operator*(const Mat4 &other) const {
            Mat4 result;
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    float sum = 0;
                    for (int k = 0; k < 4; ++k) {
                        sum += matrix[k * 4 + row] * other.matrix[col * 4 + k];
                    }
                    result.matrix[col * 4 + row] = sum;
                }
            }
            return result;
        }
    };
}

#endif //VULKAN_STARTER_APP_MATH_HPP
