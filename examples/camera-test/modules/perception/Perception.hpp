#pragma once

#include "Fastscnn.hpp"
#include "YoloV3.hpp"
#include "ProjectPaths.h"
#include "PerfProfiler.hpp"

/* Configuration of the visualization window */
#define VISUALIZATION_ENABLED true
#define VIS_WINDOW_W SEG_MAP_W
#define VIS_WINDOW_H DETECTION_ROI_H + SEG_MAP_H

/* Camera input dimensions */
#define CAMERA_INPUT_W 1920
#define CAMERA_INPUT_H 1080

#define BETWEEN_LANES 3

struct DetectedSign
{
    Yolo::Detection det;
    int miss_cnt;
    int frame_cnt = 0;
};

class Perception
{
public:
    /* Perception module constructor */
    Perception() : seg_network(segModelPath), det_network(detModelPath)
    {
        if (!cudaAllocMapped(&det_vis_image, make_int2(DETECTION_ROI_W, DETECTION_ROI_H)))
        {
            LogError("Perception: Failed to allocate CUDA memory for detection vis image\n");
        }
        if (!cudaAllocMapped(&left_lane_x_limits, 2 * IMG_HEIGHT * CONTOUR_RES * sizeof(int)))
        {
            LogError("Perception: Failed to allocate CUDA memory for left_lane_x_limits\n");
        }
        if (!cudaAllocMapped(&right_lane_x_limits, 2 * IMG_HEIGHT * CONTOUR_RES * sizeof(int)))
        {
            LogError("Perception: Failed to allocate CUDA memory for right_lane_x_limits\n");
        }
        if (!cudaAllocMapped(&charging_pad_center, 4 * sizeof(int)))
        {
            LogError("Perception: Failed to allocate CUDA memory for charging_pad_center\n");
        }
        if (!cudaAllocMapped(&obstacle_limits, 4 * sizeof(int)))
        {
            LogError("Perception: Failed to allocate CUDA memory for obstacle_limits\n");
        }
        perf_profiler_ptr = PerfProfiler::getInstance();
        InitNetworks();
    };

    /* Perception module destructor */
    ~Perception();
    /* Main function that processes the module */
    int Process(pixelType *img_input, pixelType *img_output);
    /* Get detected sign interface function */
    int GetDetection(Yolo::Detection* det);
    /* Get segmap pointer */
    uint8_t* GetSegmapPtr();
    /* Get the array that contains the x limits coordinates of left lane */
    int* GetLeftLaneXLimits();
    /* Get the array that contains the x limits coordinates of right lane */
    int* GetRightLaneXLimits();
    /*Get the charging pad center array pointer */
    int* GetChargingPadCenter();
    /* Get the obstacle limits array */
    int* GetObstacleLimitsArray();

private:
    /* Performance profiler instance pointer */
    PerfProfiler* perf_profiler_ptr;
    /* Network TensorRT objects */
    FastScnn seg_network;
    YoloV3 det_network;

    uint8_t* classmap_ptr;
    int* left_lane_x_limits;
    int* right_lane_x_limits;
    /* Charging pad limits array */
    int *charging_pad_center;
    /* Obstacle limits array */
    int *obstacle_limits;
    /* Pointer to the detection overlay image */
    pixelType *det_vis_image;
    /* Structure that contains info about the detected sign */
    DetectedSign detected_sign;
    /* Initialize the networks*/
    int InitNetworks();
    /* Get the image thats fed to the detection network before preprocessing */
    void GetDetImage(pixelType *input_img);
    /* Overlay detection image on the visualization image */
    void OverlayDetImage(pixelType *vis_img);
    /* Overlay bboxes on det image */
    void OverlayBBoxesOnVisImage(uchar3 *out_image, int img_width, int img_height);
    /* Filter the detections to get a reliable reading of the detected sign */
    void FilterDetections(std::vector<Yolo::Detection> detections);
    /* Overlay the segmap*/
    void OverlaySegImage(pixelType *img, int middle_lane_x);
    /* Initialize all the limits arrays */
    void InitializeLimitsArrays();
};