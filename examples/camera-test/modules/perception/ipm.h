#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

__global__ void rgbToIpm(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);
extern "C" void warpImageK(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);

extern "C" void OverlaySegImageK(uchar3 *img, int middle_lane_x, const uint8_t *classmap_ptr, int width, int height);