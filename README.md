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

## Dependencies

- NVIDIA GPU with CUDA capability.
- CUDA toolkit and libraries installed.
- FreeImage library for image input/output operations.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.
