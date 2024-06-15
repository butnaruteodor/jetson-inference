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
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(start_time - std::chrono::high_resolution_clock::now());

    switch (state)
    {
    /* Wait for command to start the car */
    case STOP:
        /* Car is stopped */
        speed_sp = 0;
        lateral_sp = 512;
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
        if (abs(elapsed_time.count() - wait_struct.last_time_sec) >= wait_struct.time_to_stop_sec)
        {
            wait_struct.first_time_wait = true;
            state = DRIVE;
        }
        /* Set desired speed and steering angle */
        speed_sp = 0;
        lateral_sp = 512;
        break;
    case DRIVE:
        lateral_sp = 512;
        for (int i = IMG_HEIGHT / CONTOUR_RES - 1; i > 0; i--)
        {
            if (right_lane_x[i] > 0)
            {
                lateral_sp = right_lane_x[i];
                break;
            }
        }
        speed_sp = 300;

        static bool detected = false;
        switch ((int)detected_sign.class_id)
        {
        case 0:
            wait_struct.time_to_stop_sec = 4;
            detected = true;
            break;
        case 1:
            break;
        case 2:
            break;
        case 3:
            break;
        default:
            /* When the sign goes out of view, the car passed it, perform the behaviour */
            if(detected == true)
            {
                detected = false;
                state = WAIT;
            }
            break;
        }

        break;
    /* Unkwonw state, shouldn't ever get here */
    default:
        speed_sp = 0;
        lateral_sp = 512;
        break;
    }
}

void Planning::GetPerceptionData(Perception *perception_module)
{
    perception_module->GetDetection(&detected_sign);
    left_lane_x_limits = perception_module->GetLeftLaneXLimits();
    right_lane_x_limits = perception_module->GetRightLaneXLimits();
    GetLaneCenterPoints();
}

void Planning::OverlayLanePoints(uchar3 *out_img)
{
    uchar3 color_center_left_lane = {255, 0, 0};
    uchar3 color_center_right_lane = {0, 255, 0};
    uchar3 color_left_ext = {0, 0, 0};
    uchar3 color_right_ext = {0, 0, 0};
    for (int i = 0; i < IMG_HEIGHT / CONTOUR_RES; i++)
    {
        /* Lane center points */
        if (left_lane_x[i] > 0)
        {
            // int dest_pos = i * CONTOUR_RES * IMG_WIDTH + left_lane_x[i];
            out_img[(i * CONTOUR_RES) * IMG_WIDTH + left_lane_x[i]] = color_center_left_lane;
            out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + left_lane_x[i]] = color_center_left_lane;
            out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + left_lane_x[i]] = color_center_left_lane;
        }
        /* Lane extremities points */
        out_img[(i * CONTOUR_RES) * IMG_WIDTH + left_lane_x_limits[2 * i]] = color_left_ext;
        out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + left_lane_x_limits[2 * i]] = color_left_ext;
        out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + left_lane_x_limits[2 * i]] = color_left_ext;

        out_img[(i * CONTOUR_RES) * IMG_WIDTH + left_lane_x_limits[2 * i + 1]] = color_right_ext;
        out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + left_lane_x_limits[2 * i + 1]] = color_right_ext;
        out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + left_lane_x_limits[2 * i + 1]] = color_right_ext;
        if (right_lane_x[i] > 0)
        {
            // int dest_pos = i * CONTOUR_RES * IMG_WIDTH + left_lane_x[i];
            out_img[(i * CONTOUR_RES) * IMG_WIDTH + right_lane_x[i]] = color_center_right_lane;
            out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + right_lane_x[i]] = color_center_right_lane;
            out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + right_lane_x[i]] = color_center_right_lane;
        }
        /* Lane extremities points */
        out_img[(i * CONTOUR_RES) * IMG_WIDTH + right_lane_x_limits[2 * i]] = color_left_ext;
        out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + right_lane_x_limits[2 * i]] = color_left_ext;
        out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + right_lane_x_limits[2 * i]] = color_left_ext;

        out_img[(i * CONTOUR_RES) * IMG_WIDTH + right_lane_x_limits[2 * i + 1]] = color_right_ext;
        out_img[(i * CONTOUR_RES + 1) * IMG_WIDTH + right_lane_x_limits[2 * i + 1]] = color_right_ext;
        out_img[(i * CONTOUR_RES + 2) * IMG_WIDTH + right_lane_x_limits[2 * i + 1]] = color_right_ext;
    }
}

int Planning::GetLateralSetpoint()
{
    return lateral_sp;
}

int Planning::GetLongitudinalSetpoint()
{
    return speed_sp;
}

void Planning::GetLaneCenterPoints()
{
    for (int i = 0; i < IMG_HEIGHT / CONTOUR_RES; i++)
    {
        /************************LEFT LANE*******************************/
        // -1 means there is no center point for the lane at this height
        left_lane_x[i] = -1;
        /* It means the two extremities of the lane are the init values, the lane is not visible at this height */
        if (left_lane_x_limits[2 * i] > left_lane_x_limits[2 * i + 1])
        {
        }
        /* We have lane extremities could be good or not good */
        else
        {
            int lane_width = left_lane_x_limits[2 * i + 1] - left_lane_x_limits[2 * i];
            /* If its the expected general width, best case scenario(the lane is probably fully visible and straight) */
            if (lane_width > LANE_WIDTH_BETWEEN_MIN && lane_width < LANE_WIDTH_BETWEEN_MAX)
            {
                left_lane_x[i] = (left_lane_x_limits[2 * i + 1] + left_lane_x_limits[2 * i]) / 2;
            }
            /* Lane width is too big, could be false classification of pixels or the direction of the lane is parallel to the camera */
            else if (lane_width > LANE_WIDTH_BETWEEN_MAX)
            {
                /* Will be estimated based on neighbours */
                left_lane_x[i] = -2;
            }
            /* Only a small part of the lane is visible, could be because of the trapezoidal field of view */
            else if (lane_width < LANE_WIDTH_BETWEEN_MIN)
            {
                /* Will be estimated based on neighbours */
                left_lane_x[i] = -3;
            }
        }

        /************************RIGHT LANE*******************************/
        // -1 means there is no center point for the lane at this height
        right_lane_x[i] = -1;
        /* It means the two extremities of the lane are the init values, the lane is not visible at this height */
        if (right_lane_x_limits[2 * i] > right_lane_x_limits[2 * i + 1])
        {
        }
        /* We have lane extremities could be good or not good */
        else
        {
            int lane_width = right_lane_x_limits[2 * i + 1] - right_lane_x_limits[2 * i];
            /* If its the expected general width, best case scenario(the lane is probably fully visible and straight) */
            if (lane_width > LANE_WIDTH_BETWEEN_MIN && lane_width < LANE_WIDTH_BETWEEN_MAX)
            {
                right_lane_x[i] = (right_lane_x_limits[2 * i + 1] + right_lane_x_limits[2 * i]) / 2;
            }
            /* Lane width is too big, could be false classification of pixels or the direction of the lane is parallel to the camera */
            else if (lane_width > LANE_WIDTH_BETWEEN_MAX)
            {
                /* Will be estimated based on neighbours */
                right_lane_x[i] = -2;
            }
            /* Only a small part of the lane is visible, could be because of the trapezoidal field of view */
            else if (lane_width < LANE_WIDTH_BETWEEN_MIN)
            {
                /* Will be estimated based on neighbours */
                right_lane_x[i] = -3;
            }
        }
    }
    /* Estimation time */
    for (int i = 0; i < IMG_HEIGHT / CONTOUR_RES; i++)
    {
        /************************LEFT LANE*******************************/
        /* wide lane */
        if (left_lane_x[i] == -2)
        {
            if (i > 1 && i < IMG_HEIGHT / CONTOUR_RES - 1)
            {
            }
        }
        /* narrow lane */
        else if (left_lane_x[i] == -3)
        {
            /* only part of the lane width is visible, estimate where the middle is based on known lane width */
            int right_most_x = left_lane_x_limits[2 * i + 1];
            /* if its negative(the lane is in the top corner somewhere and only a small part is visible) we dont care will be ignored */
            left_lane_x[i] = (right_most_x + (right_most_x - LANE_WIDTH)) / 2;
        }

        /************************RIGHT LANE*******************************/

        /* wide lane */
        if (right_lane_x[i] == -2)
        {
            if (i > 1 && i < IMG_HEIGHT / CONTOUR_RES - 1)
            {
            }
        }
        /* narrow lane */
        else if (right_lane_x[i] == -3)
        {
            /* only part of the lane width is visible, estimate where the middle is based on known lane width */
            int left_most_x = right_lane_x_limits[2 * i];
            /* Here the coordinate could be higher than the width of the image, we need to clamp */
            right_lane_x[i] = (left_most_x + (left_most_x + LANE_WIDTH)) / 2;
            if (right_lane_x[i] >= IMG_WIDTH)
            {
                right_lane_x[i] = IMG_WIDTH - 1;
            }
        }
    }
}