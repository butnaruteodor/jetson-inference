#include "Planning.hpp"
#include <chrono>

Planning::Planning()
{
    state = WAIT;
    planning_vars.first_time_wait = true;
}

void Planning::RunStateHandler()
{
    static auto duration = duration_cast<seconds>(high_resolution_clock::now());

    switch (state)
    {
    /* The car will stop in this state for a planning_vars.time_to_stop_sec seconds */
    case WAIT:
        /* Timestamp for the first time the state machine enters the wait state */
        if (planning_vars.first_time_wait == true)
        {
            planning_vars.last_time_sec = duration.count();
            planning_vars.first_time_wait = false;
        }
        /* If planning_vars.time_to_stop seconds has passed the car stopped enough */
        if (duration.count() - planning_vars.last_time_sec >= planning_vars.time_to_stop_sec)
        {
            planning_vars.first_time_wait = true;
            state = DRIVE;
        }

        /* Set desired speed and steering angle */
        desired_speed = 0;
        desired_steering_angle = 0;
        break;
    case DRIVE:
    
        break;
    case STOP:
        break;
    default:
        break;
    }
}
