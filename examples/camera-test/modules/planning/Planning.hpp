#pragma once
#include "Perception.hpp"

class Planning
{

public:
    Planning();
    ~Planning();
    void RunStateHandler();
    /* Get the detected sign and the pointer to the segmap */
    void GetPerceptionData(Perception* perception_module);

private:
    enum State
    {
        WAIT,
        DRIVE,
        STOP
    };
    struct WaitStruct
    {
        int last_time_sec;
        int time_to_stop_sec;
        bool first_time_wait;
    };

    State state;
    int desired_speed;
    int desired_steering_angle;
    uint8_t *segmap_ptr;

    Yolo::Detection detected_sign;
    WaitStruct wait_struct;
};