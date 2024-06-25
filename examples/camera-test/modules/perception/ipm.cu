#include "ipm.h"

#define UV_GRID_COLS 524288
#define OUT_IMAGE_WIDTH 1024
#define OUT_IMAGE_HEIGHT 512

__global__ void rgbToIpm(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height - 1)
        return;

    int uvIndex = y * width + x;

    int ui = uGrid[uvIndex];
    int vi = vGrid[uvIndex];

    int inIndex = vi * 1920 + ui;
    int outIndex = y * width + x;

    if (ui >= 0 && ui < 1920 && vi >= 0 && vi < 1080)
    {

        output[outIndex] = (input[inIndex].x / 255.0f - 0.485f) / 0.229f;
        output[outIndex + 524288] = (input[inIndex].y / 255.0f - 0.456f) / 0.224f;
        output[outIndex + 1048576] = (input[inIndex].z / 255.0f - 0.406f) / 0.225f;
    }
    else
    {
        output[outIndex] = -2.11790393f;
        output[outIndex + 524288] = -2.035714286f;
        output[outIndex + 1048576] = -1.804444444f;
    }
}
void warpImageK(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height)
{
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((OUT_IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (OUT_IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    rgbToIpm<<<gridDim, blockDim>>>(input, output, uGrid, vGrid, 1024, 512);
}

__global__ void OverlaySegImageKernel(uchar3 *img, int middle_lane_x, const uint8_t *classmap_ptr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        if ((int)classmap_ptr[index] == 0) {
            img[index].x = 128;
            img[index].y = 64;
            img[index].z = 128;
        }
        else if ((int)classmap_ptr[index] == 1) {
            img[index].x = 244;
            img[index].y = 35;
            img[index].z = 232;
        }
        else if ((int)classmap_ptr[index] == 2) {
            img[index].x = 70;
            img[index].y = 70;
            img[index].z = 70;
        }
        else if ((int)classmap_ptr[index] == 3) {
            img[index].x = 102;
            img[index].y = 102;
            img[index].z = 156;
        }
        else if ((int)classmap_ptr[index] == 4) {
            img[index].x = 190;
            img[index].y = 153;
            img[index].z = 153;
        }
        else if ((int)classmap_ptr[index] == 5) {
            img[index].x = 255;
            img[index].y = 255;
            img[index].z = 0;
        }
        if (x == middle_lane_x) {
            img[index].x = 0;
            img[index].y = 255;
            img[index].z = 0;
        }
    }
}

void OverlaySegImageK(uchar3 *img, int middle_lane_x, const uint8_t *classmap_ptr, int width, int height) {
    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    OverlaySegImageKernel<<<gridSize, blockSize>>>(img, middle_lane_x, classmap_ptr, width, height);
}

__global__ void OverlayDetImageKernel(uchar3* img_output, uchar3* det_vis_image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < DETECTION_ROI_W && y < DETECTION_ROI_H)
    {
        int srcPos = y * DETECTION_ROI_W + x;
        int dstPos = (y + 512) * 1024 + x;

        // Copy pixel
        img_output[dstPos] = det_vis_image[srcPos];
    }
}

void OverlayDetImagek(uchar3* img_output, uchar3* det_vis_image)
{
    // Define the grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size as needed
    dim3 gridDim((DETECTION_ROI_W + blockDim.x - 1) / blockDim.x, (DETECTION_ROI_H + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    OverlayDetImageKernel<<<gridDim, blockDim>>>(img_output, det_vis_image);
}