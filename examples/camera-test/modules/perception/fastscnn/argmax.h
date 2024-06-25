#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define IMG_WIDTH 1024
#define IMG_HEIGHT 512
#define IMG_HW 524288
#define NUM_CLASSES 6
#define CONTOUR_RES 50

extern "C" void generateClassMap(float *output_1D, uint8_t *class_map, int* left_lane_x_limits, int* right_lane_x_limits, int *charging_pad_center, int* obstacle_limits);
