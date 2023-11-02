#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void sobel_filter(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        float gx = input[(y-1)*width + (x-1)] - input[(y-1)*width + (x+1)] 
                 + 2 * input[y*width + (x-1)] - 2 * input[y*width + (x+1)] 
                 + input[(y+1)*width + (x-1)] - input[(y+1)*width + (x+1)];

        float gy = input[(y-1)*width + (x-1)] + 2 * input[(y-1)*width + x] + input[(y-1)*width + (x+1)] 
                 - input[(y+1)*width + (x-1)] - 2 * input[(y+1)*width + x] - input[(y+1)*width + (x+1)];

        float magnitude = sqrt(gx*gx + gy*gy);

        // Bound the values between 0 and 255
        magnitude = magnitude > 255 ? 255 : magnitude;
        magnitude = magnitude < 0 ? 0 : magnitude;

        output[y*width + x] = (unsigned char)magnitude;
    }
}

// PPM image I/O functions (only for grayscale images here)
bool read_image(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return false;

    std::string header;
    input >> header;
    if (header != "P6") return false;

    input >> width >> height;
    int max_val;
    input >> max_val;
    input.get(); // consume newline

    data.resize(width * height * 3); // 3 for RGB
    input.read(reinterpret_cast<char*>(data.data()), width * height * 3);

    // Convert to grayscale
    for (int i = 0; i < width * height; i++) {
        unsigned char gray = (data[3*i] + data[3*i+1] + data[3*i+2]) / 3;
        data[i] = gray;
    }

    data.resize(width * height);
    return true;
}

bool write_image(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) return false;

    output << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        output << data[i] << data[i] << data[i]; // grayscale as RGB
    }
    return true;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./sobel_filter <input_image_path.ppm> <output_image_path.ppm>" << std::endl;
        return -1;
    }

    std::vector<unsigned char> image, output;
    int width, height;
    if (!read_image(argv[1], image, width, height)) {
        std::cerr << "Error reading image!" << std::endl;
        return -1;
    }
    output.resize(width * height);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, image.data(), width * height, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sobel_filter<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(output.data(), d_output, width * height, cudaMemcpyDeviceToHost);

    if (!write_image(argv[2], output, width, height)) {
        std::cerr << "Error writing output image!" << std::endl;
        return -1;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

