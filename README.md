# CUDA Sobel Filter Project
This is a project demonstrating the implementation of the Sobel filter using CUDA, created as part of the "CUDA at Scale for the Enterprise" Coursera course.

## Project Description

The Sobel filter, or Sobel operator, is used in image processing primarily for edge detection. The operator uses two 3×3 convolution kernels which are applied to the original image to produce separate measurements of the gradient in the x and y directions.

This CUDA-based implementation speeds up the Sobel filter processing by leveraging the parallel processing capabilities of NVIDIA GPUs.

The application inputs a greyscale N by M image and outputs an N by M image (in PPM format) highlighting the edges detected in the original image.

## Usage

1. Compile the program using the provided Makefile.
2. Run the program by providing a path to a greyscale image:
3. The program will output a new PPM image with edges highlighted.

## Example
For a demonstration, you can run the provided `run.sh` script which applies the Sobel filter to a sample greyscale image.

Input and output examples can be found in `data/panda.ppm` and `data/panda-sobel.ppm`, respectively.

<p float="left">
  <img src="https://github.com/zzh237/cuda_enterprise/blob/main/data/panda.png" width="45%" />
  <img src="https://github.com/zzh237/cuda_enterprise/blob/main/data/panda-sobel.png" width="45%" /> 
</p>

## Dependencies

- NVIDIA GPU with CUDA capability.
- CUDA toolkit and libraries installed.
- FreeImage library for image input/output operations.

