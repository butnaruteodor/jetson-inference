#pragma once
#include "Perception.hpp"

class Planning
{

public:
    Planning();
    ~Planning();
    void RunStateHandler();
    

private:
    enum State
    {
        WAIT,
        DRIVE,
        STOP
    };
    struct PlanningVars
    {
        int last_time_sec;
        int time_to_stop_sec;
        bool first_time_wait;
    }

    State state;
    int desired_speed;
    int desired_steering_angle;
    DetectedSign detected_sign;
    PlanningVars planning_vars;
};