#pragma once

#include "Fastscnn.hpp"
#include "YoloV3.hpp"
#include "ProjectPaths.h"

/* Configuration of the visualization window */
#define VISUALIZATION_ENABLED true
#define VIS_WINDOW_W SEG_MAP_W
#define VIS_WINDOW_H DETECTION_ROI_H + SEG_MAP_H

/* Camera input dimensions */
#define CAMERA_INPUT_W 1920
#define CAMERA_INPUT_H 1080

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
        if (!cudaAllocMapped(&charging_pad_center, 2 * sizeof(int)))
        {
            LogError("Perception: Failed to allocate CUDA memory for charging_pad_center\n");
        }
    };

    /* Perception module destructor */
    ~Perception();

    /* Initialize the module*/
    int InitModule();

    /* Main function that processes the module */
    int RunPerception(pixelType *img_input, pixelType *img_output);

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

private:
    /* Network TensorRT objects */
    FastScnn seg_network;
    YoloV3 det_network;

    uint8_t* classmap_ptr;
    int* left_lane_x_limits;
    int* right_lane_x_limits;

    /* Charging pad array */
    int *charging_pad_center;

    /* Pointer to the detection overlay image */
    pixelType *det_vis_image;

    /* Structure that contains info about the detected sign */
    DetectedSign detected_sign;

    /* Get the image thats fed to the detection network before preprocessing */
    void GetDetImage(pixelType *input_img);

    /* Overlay detection image on the visualization image */
    void OverlayDetImage(pixelType *vis_img);

    /* Filter the detections to get a reliable reading of the detected sign */
    void FilterDetections(std::vector<Yolo::Detection> detections);

    void OverlaySegImage(pixelType *img, int middle_lane_x);

    /* Initialize the lane limits array */
    void InitializeLaneLimitsArray(int* lane_x_limits);

    void InitializePadArray(int* pad_center);
};