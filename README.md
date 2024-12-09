# CUDA Raytracer

To use the raytracer, download the directory and run `main.exe` with the following input:

main.exe scene.txt > output.ppm

An example scene format is provided in the file: `r3_spheres_perspective.txt`.

**OUTPUT IMAGE HERE**: `example_output.png`

---

## Description

This repository was created based on:
- The first chapter of the raytracing in a week course for the C++ code.
- The following tutorial on how to optimize it with CUDA:
  - [Ray Tracing in One Weekend](https://raytracing.github.io/)
  - [NVIDIA Blog: Accelerated Ray Tracing in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)

The code has been modified to handle:
- Perspective camera settings.
- Dynamic scene loading from `.txt` files.

---

## Planned Additions

- Different types of intersections.
- New materials.
