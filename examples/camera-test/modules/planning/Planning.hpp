#pragma once
#include "Perception.hpp"

/* The lane width in pixels is between LANE_WIDTH_BETWEEN_MIN and LANE_WIDTH_BETWEEN_MAX */
#define LANE_WIDTH_BETWEEN_MIN 150
#define LANE_WIDTH_BETWEEN_MAX 300

/* Aproximate value of the width of the lane in pixels */
#define LANE_WIDTH 200

class Planning
{

public:
    Planning();
    ~Planning();
    void RunStateHandler();
    /* Get the detected sign and the pointer to the segmap */
    void GetPerceptionData(Perception *perception_module);
    /* Overlay lane center on image */
    void OverlayLanePoints(uchar3 *out_img);
    /* Get the lateral setpoint */
    int GetLateralSetpoint();
    /* Get the longitudinal setpoint */
    int GetLongitudinalSetpoint();

private:
    enum State
    {
        WAIT,
        DRIVE,
        STOP,
        PARK
    };
    struct WaitStruct
    {
        int last_time_sec;
        int time_to_stop_sec;
        bool first_time_wait;
    };

    State state;
    State last_state;
    int speed_sp;
    int lateral_sp;

    Yolo::Detection detected_sign;
    WaitStruct wait_struct;

    /* Lane extremities x values, taken raw from perception module */
    int *left_lane_x_limits;
    int *right_lane_x_limits;
    /* Charging pad center*/
    int *charging_pad_center;
    /* Obstacle limits array */
    int *obstacle_limits;
    /* Right lane center x values */
    int right_lane_x[OUT_IMG_H / CONTOUR_RES];
    /* Left lane center x values */
    int left_lane_x[OUT_IMG_H / CONTOUR_RES];

    /* Pre processes the data from the perception module, gives the lane center points */
    void GetLaneCenterPoints();
    /* Get lane lateral setpoint by lane id */
    int GetLaneCenter(int lane_idx, bool obstacle);
    /* Choose the free lane */
    bool IsObstacleOnLane();
};