#include "argmax.h"

__global__ void generateClassMapKernel(float *output_1D, uint8_t *class_map, int *left_lane_x_limits, int *right_lane_x_limits)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // equivalent of i in your original function
    int j = blockIdx.x * blockDim.x + threadIdx.x; // equivalent of j in your original function

    if (i < IMG_HEIGHT && j < IMG_WIDTH)
    {
        float max_value = -10000.0f;
        int max_class = 0;
        int i_width = i * IMG_WIDTH;
        int i_width_j = i_width + j;
        int upmost_i = -1;

        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            int index = c * IMG_HW + i_width_j;
            if (output_1D[index] > max_value)
            {
                max_value = output_1D[index];
                max_class = c;
            }
        }
        class_map[i_width_j] = max_class;

        /* Process only the sampling lines pixels */
        if (i % CONTOUR_RES == 0)
        {
            // Calculate the base index for the class and y-coordinate
            int base_index = i / CONTOUR_RES * 2;

            if (max_class == 2)
            {
                // Use atomicMax and atomicMin to safely update the maximum and minimum x-coordinates
                atomicMin(&left_lane_x_limits[base_index], j);
                atomicMax(&left_lane_x_limits[base_index + 1], j);
            }
            else if(max_class == 1)
            {
                // Use atomicMax and atomicMin to safely update the maximum and minimum x-coordinates
                atomicMin(&right_lane_x_limits[base_index], j);
                atomicMax(&right_lane_x_limits[base_index + 1], j);
            }
        }
    }
}
void generateClassMap(float *output_1D, uint8_t *class_map, int *left_lane_x_limits, int *right_lane_x_limits)
{

    dim3 blockDim(16, 16); // You might need to tune these numbers
    dim3 gridDim((IMG_WIDTH + blockDim.x - 1) / blockDim.x, (IMG_HEIGHT + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    generateClassMapKernel<<<gridDim, blockDim>>>(output_1D, class_map, left_lane_x_limits, right_lane_x_limits);
}