#include "argmax.h"

__global__ void initializeClassPixelsIndicesK(int *class_pixels_indices)
{
    int class_id = blockIdx.x;
    int y = blockIdx.y;

    // Calculate the base index for the given class and y-coordinate
    int base_index = (class_id * IMG_HEIGHT + y) * (IMG_WIDTH + 1);

    // Initialize the position counter to 1
    class_pixels_indices[base_index] = 1;
}

__global__ void generateClassMapKernel(float *output_1D, uint8_t *class_map, int *class_pixels_indices)
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

        if (max_class == 2)
        {
            if(i%CONTOUR_RES==0){
                // Calculate the base index for the class and y-coordinate
                int base_index = i/CONTOUR_RES * 2;

                // Use atomicMax and atomicMin to safely update the maximum and minimum x-coordinates
                atomicMin(&class_pixels_indices[base_index], j);
                atomicMax(&class_pixels_indices[base_index + 1], j);
        }
            }
            
    }
}
void generateClassMap(float *output_1D, uint8_t *class_map, int *class_pixels_indices)
{

    dim3 blockDim(16, 16); // You might need to tune these numbers
    dim3 gridDim((IMG_WIDTH + blockDim.x - 1) / blockDim.x, (IMG_HEIGHT + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    generateClassMapKernel<<<gridDim, blockDim>>>(output_1D, class_map, class_pixels_indices);
}

void initializeClassPixelsIndices(int *class_pixels_indices)
{
    // Define block and grid sizes for the initialization kernel
    dim3 initBlockDim(1, 1);
    dim3 initGridDim(IMG_HEIGHT/CONTOUR_RES, 2);

    // Launch the initialization kernel
    initializeClassPixelsIndicesK<<<initGridDim, initBlockDim>>>(class_pixels_indices);
}