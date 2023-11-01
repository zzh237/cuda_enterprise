#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void sobel_filter(unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        float gx = input[(y-1)*width + (x-1)] - input[(y-1)*width + (x+1)] 
                 + 2 * input[y*width + (x-1)] - 2 * input[y*width + (x+1)] 
                 + input[(y+1)*width + (x-1)] - input[(y+1)*width + (x+1)];

        float gy = input[(y-1)*width + (x-1)] + 2 * input[(y-1)*width + x] + input[(y-1)*width + (x+1)] 
                 - input[(y+1)*width + (x-1)] - 2 * input[(y+1)*width + x] - input[(y+1)*width + (x+1)];

        output[y*width + x] = sqrt(gx*gx + gy*gy);
    }
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./sobel_filter <input_image_path> <output_image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if(!image.data) {
        std::cerr << "Error reading image!" << std::endl;
        return -1;
    }

    cv::Mat output(image.rows, image.cols, CV_8UC1);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, image.rows * image.cols);
    cudaMalloc(&d_output, image.rows * image.cols);

    cudaMemcpy(d_input, image.data, image.rows * image.cols, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((image.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (image.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sobel_filter<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, image.cols, image.rows);

    cudaMemcpy(output.data, d_output, image.rows * image.cols, cudaMemcpyDeviceToHost);

    cv::imwrite(argv[2], output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
