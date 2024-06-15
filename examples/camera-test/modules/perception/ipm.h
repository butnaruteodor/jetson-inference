#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include "YoloV3.hpp"

__global__ void rgbToIpm(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);
extern "C" void warpImageK(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);

extern "C" void OverlaySegImageK(uchar3 *img, int middle_lane_x, const uint8_t *classmap_ptr, int width, int height);

extern "C" void OverlayDetImagek(uchar3* img_output, uchar3* det_vis_image);