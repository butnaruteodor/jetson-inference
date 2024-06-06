#include "Planning.hpp"
#include <chrono>

Planning::Planning()
{
    state = STOP;
    wait_struct.first_time_wait = true;
}

Planning::~Planning()
{
}

void Planning::RunStateHandler()
{
    auto static start_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(start_time-std::chrono::high_resolution_clock::now());

    switch (state)
    {
    /* Wait for command to start the car */
    case STOP:
        /* Car is stopped */
        desired_speed = 0;
        desired_steering_angle = 0;
        state = DRIVE;
        break;
    /* The car will stop in this state for a planning_vars.time_to_stop_sec seconds */
    case WAIT:
        /* Timestamp for the first time the state machine enters the wait state */
        if (wait_struct.first_time_wait == true)
        {
            wait_struct.last_time_sec = elapsed_time.count();
            wait_struct.first_time_wait = false;
        }
        /* If planning_vars.time_to_stop seconds has passed the car stopped enough */
        if (elapsed_time.count() - wait_struct.last_time_sec >= wait_struct.time_to_stop_sec)
        {
            wait_struct.first_time_wait = true;
            state = DRIVE;
        }

        /* Set desired speed and steering angle */
        desired_speed = 0;
        desired_steering_angle = 0;
        break;
    case DRIVE:
        LogInfo("Sign: %f\n",detected_sign.class_id);
        break;
    /* Unkwonw state, shouldn't ever get here */
    default:
        desired_speed = 0;
        desired_steering_angle = 0;
        break;
    }
}

void Planning::GetPerceptionData(Perception* perception_module)
{
    perception_module->GetDetection(&detected_sign);
    segmap_ptr = perception_module->GetSegmapPtr();
}
