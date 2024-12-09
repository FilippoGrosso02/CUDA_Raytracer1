#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
#include <cmath> // For `tan` and `M_PI`

class camera {
public:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    __device__ camera() {
        // Default camera setup
        lower_left_corner = vec3(-2.0, -1.0, -1.0);
        horizontal = vec3(4.0, 0.0, 0.0);
        vertical = vec3(0.0, 2.0, 0.0);
        origin = vec3(0.0, 0.0, 0.0);
    }

    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
        // Custom camera setup
        origin = lookfrom;
        float theta = vfov * 3.14f / 180.0f; // Convert vertical FOV to radians
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);
        lower_left_corner = origin - half_width * u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    __device__ ray get_ray(float u, float v) const {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
};

#endif
