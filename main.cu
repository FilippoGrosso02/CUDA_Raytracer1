#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"

// CUDA error checking macro
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
            << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

struct SphereData {
    vec3 center;
    float radius;
};

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 color(const ray& r, hitable** world, vec3 backgroundColor, curandState* local_rand_state) {
    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target - rec.p);
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * backgroundColor;
            return cur_attenuation * c;
        }
    }
    return vec3(0.1, 0.1, 0.1);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, vec3 backgroundColor, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, backgroundColor, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, SphereData* spheres, int num_spheres, vec3 camera_origin, vec3 camera_lookat, vec3 camera_up, float vfov, float aspect) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaMalloc((void**)&d_list, num_spheres * sizeof(hitable*));
        for (int i = 0; i < num_spheres; i++) {
            d_list[i] = new sphere(spheres[i].center, spheres[i].radius);
        }
        *d_world = new hitable_list(d_list, num_spheres);
        *d_camera = new camera(camera_origin, camera_lookat, camera_up, vfov, aspect);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera, int num_spheres) {
    for (int i = 0; i < num_spheres; i++) {
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

void loadSceneFromFile(const std::string& filename, std::vector<SphereData>& spheres, vec3& backgroundColor, vec3& camera_origin, vec3& camera_lookat, vec3& camera_up, float& vfov) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "PerspectiveCamera") {
            std::cerr << "Loading Camera settings...\n";
            while (std::getline(file, line)) {
                std::istringstream cameraSS(line);
                cameraSS >> token;
                if (token == "center") {
                    cameraSS >> camera_origin[0] >> camera_origin[1] >> camera_origin[2];
                    std::cerr << "  Camera origin: " << camera_origin << "\n";
                }
                else if (token == "direction") {
                    cameraSS >> camera_lookat[0] >> camera_lookat[1] >> camera_lookat[2];
                    std::cerr << "  Camera lookat: " << camera_lookat << "\n";
                }
                else if (token == "up") {
                    cameraSS >> camera_up[0] >> camera_up[1] >> camera_up[2];
                    std::cerr << "  Camera up vector: " << camera_up << "\n";
                }
                else if (token == "angle") {
                    cameraSS >> vfov;
                    std::cerr << "  Camera vfov: " << vfov << "\n";
                }
                else if (token == "}") {
                    std::cerr << "  End of Camera settings.\n";
                    break;
                }
            }
        }
        else if (token == "Background") {
            std::cerr << "Loading Background color...\n";
            std::string dummy;
            ss >> dummy; // Skip `{`
            float r, g, b;
            ss >> token >> r >> g >> b; // Read `color` and values
            backgroundColor = vec3(r, g, b);
            std::cerr << "  Background color: " << backgroundColor << "\n";
        }
        else if (token == "Sphere") {
            std::cerr << "Loading Sphere...\n";
            vec3 center;
            float radius;
            while (std::getline(file, line)) {
                std::istringstream sphereSS(line);
                sphereSS >> token;
                if (token == "center") {
                    sphereSS >> center[0] >> center[1] >> center[2];
                    std::cerr << "  Sphere center: " << center << "\n";
                }
                else if (token == "radius") {
                    sphereSS >> radius;
                    std::cerr << "  Sphere radius: " << radius << "\n";
                }
                else if (token == "}") {
                    std::cerr << "  End of Sphere.\n";
                    break;
                }
            }
            spheres.push_back({ center, radius });
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene_file>\n";
        return 1;
    }

    std::string scene_file = argv[1];

    // Initialize required variables
    const int nx = 1200; // Image width
    const int ny = 600;  // Image height
    const int ns = 100;  // Number of samples per pixel
    const int tx = 8;    // Threads per block (X)
    const int ty = 8;    // Threads per block (Y)

    vec3 backgroundColor;       // Background color
    vec3 camera_origin, camera_lookat, camera_up;
    float vfov;                 // Camera vertical FOV
    std::vector<SphereData> spheres; // Vector to store sphere data

    // Load scene from file
    loadSceneFromFile(scene_file, spheres, backgroundColor, camera_origin, camera_lookat, camera_up, vfov);

    // Number of pixels in the image
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate frame buffer memory
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Allocate memory for random states
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // Allocate memory for hitable objects and camera
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, spheres.size() * sizeof(hitable*)));

    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    // Allocate GPU memory for spheres and copy from host to device
    SphereData* d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, spheres.size() * sizeof(SphereData)));
    checkCudaErrors(cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice));

    // Create the world using CUDA kernel
    float aspect = float(nx) / float(ny);
    create_world << <1, 1 >> > (d_list, d_world, d_camera, d_spheres, spheres.size(), camera_origin, camera_lookat, camera_up, vfov, aspect);
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize random states
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty); // Ensure full coverage
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    // Render the scene
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, backgroundColor, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    // Output the image in PPM format
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // Free allocated memory
    free_world << <1, 1 >> > (d_list, d_world, d_camera, spheres.size());
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_spheres));
    checkCudaErrors(cudaFree(d_list));

    // Reset the device
    cudaDeviceReset();
    return 0;
}
