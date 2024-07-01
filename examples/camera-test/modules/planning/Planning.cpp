#include "Planning.hpp"
#include <chrono>

/* When the height size is above the threshold the sign is considered to be near the car */
#define SIGN_HEIGHT_SIZE_THRESH 135
/* How often a sign will be considered in seconds */
#define SIGN_DETECTION_FREQ_THRESH 5
#define LIGHT_SIZE_THRESH 140
#define LIGHT_CREEP_SIZE_THRESH 80
/* Distance until the car starts creeping to the pad */
#define PAD_DISTANCE_THRESHOLD 200
/* Distance from the obstacle until the car changes lane */
#define OBSTACLE_DISTANCE_THRESH 415
/* Car speeds */
#define SLOW_SPEED 15
#define NORMAL_SPEED 30

int last_det_time[4];

static bool IsSignClose(Yolo::Detection det, std::chrono::seconds elapsed_time)
{
    /* The closer the sign is the bigger the y axis center coordinate of the bbox will be,
     * the width is also considered because when it is below a certain threshold the sign is going out of view
     * traffic light is ignored because different logic
     */
    if ((det.bbox[3] > SIGN_HEIGHT_SIZE_THRESH) &&
        (abs(elapsed_time.count() - last_det_time[(int)det.class_id]) > SIGN_DETECTION_FREQ_THRESH))
    {
        last_det_time[(int)det.class_id] = elapsed_time.count();
        return true;
    }

    return false;
}

Planning::Planning()
{
    state = STOP;
    wait_struct.first_time_wait = true;
    perf_profiler_ptr = PerfProfiler::getInstance();
}

Planning::~Planning()
{
}
void Planning::RunStateHandler()
{
    auto static start_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(start_time - std::chrono::high_resolution_clock::now());
    static bool need_to_creep = true;
    static bool is_green_light = false;

    /* Process detected sign */
    switch ((int)detected_sign.class_id)
    {
    case YoloV3::GREEN_LIGHT_LABEL_ID:
        is_green_light = true;
        need_to_creep = false;
        break;
    default:
        break;
    }
    LogInfo("State: %d\n", (int)state);
    switch (state)
    {
    /* Wait for command to start the car */
    case STOP:
        /* Car is stopped */
        speed_sp = 0;
        lateral_sp = 512;
        state = DRIVE;
        break;
    /* The car will stop in this state for a wait_struct.time_to_stop_sec seconds */
    case WAIT:
        /* Timestamp for the first time the state machine enters the wait state */
        if (wait_struct.first_time_wait == true)
        {
            wait_struct.last_time_sec = elapsed_time.count();
            wait_struct.first_time_wait = false;
        }
        /* If wait_struct.time_to_stop seconds has passed the car stopped enough or the light turned green */
        if ((wait_struct.time_to_stop_sec > 0) && (abs(elapsed_time.count() - wait_struct.last_time_sec) >= wait_struct.time_to_stop_sec) ||
            (wait_struct.time_to_stop_sec < 0) && (is_green_light))
        {
            wait_struct.first_time_wait = true;
            state = DRIVE;
        }
        /* Set desired speed */
        speed_sp = 0;
        break;
    case DRIVE:
    {
        if (IsObstacleOnLane())
        {
            lateral_sp = GetLaneCenter(BETWEEN_LANES);
        }
        else
        {
            lateral_sp = GetLaneCenter(RIGHT_LANE);
        }

        if (need_to_creep == true)
        {
            speed_sp = SLOW_SPEED;
        }
        else
        {
            speed_sp = NORMAL_SPEED;
        }
        if (detected_sign.class_id == YoloV3::PARK_SIGN_LABEL_ID)
        {
            if (IsSignClose(detected_sign, elapsed_time))
            {
                need_to_creep = true;
                wait_struct.time_to_stop_sec = 5;
                state = WAIT;
            }
        }
        else if (detected_sign.class_id == YoloV3::CHARGE_SIGN_LABEL_ID)
        {
            if (IsSignClose(detected_sign, elapsed_time))
                state = PARK;
        }
        else if ((detected_sign.class_id == YoloV3::STOP_SIGN_LABEL_ID) || (detected_sign.class_id == YoloV3::CROSS_SIGN_LABEL_ID))
        {
            if (IsSignClose(detected_sign, elapsed_time))
            {
                if (detected_sign.class_id == YoloV3::STOP_SIGN_LABEL_ID)
                {
                    wait_struct.time_to_stop_sec = 4;
                }
                else
                {
                    wait_struct.time_to_stop_sec = 2;
                }
                state = WAIT;
            }
        }
        else if ((detected_sign.class_id == YoloV3::YELLOW_LIGHT_LABEL_ID) || (detected_sign.class_id == YoloV3::RED_LIGHT_LABEL_ID))
        {
            if (detected_sign.bbox[3] > LIGHT_SIZE_THRESH && need_to_creep)
            {
                is_green_light = false;
                wait_struct.time_to_stop_sec = -1;
                state = WAIT;
            }
        }
        break;
    }
    case PARK:
        static bool passed_pad = false;
        static int time_to_adjust = elapsed_time.count();
        speed_sp = NORMAL_SPEED;
        lateral_sp = GetLaneCenter(RIGHT_LANE);
        /* If charging pad was detected */
        if (charging_pad_center[0] < charging_pad_center[1])
        {
            if ((charging_pad_center[2] + charging_pad_center[3]) / 2 > PAD_DISTANCE_THRESHOLD)
            {
                lateral_sp = charging_pad_center[1] - 20;
                speed_sp = 10;
                passed_pad = true;
                time_to_adjust = elapsed_time.count();
            }
        }
        else
        {
            if (abs(elapsed_time.count() - time_to_adjust) > 0)
            {
                if (passed_pad)
                {
                    passed_pad = false;
                    wait_struct.time_to_stop_sec = 6;
                    state = WAIT;
                }
            }
            else
            {
                speed_sp = 10;
            }
        }

        break;
    /* Unknown state, shouldn't ever get here */
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
    charging_pad_center = perception_module->GetChargingPadCenter();
    obstacle_limits = perception_module->GetObstacleLimitsArray();
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

void Planning::Process(Perception *perception_module)
{
    perf_profiler_ptr->PROFILER_BEGIN_CPU(PROFILER_PLANNING);
    GetPerceptionData(perception_module);
    RunStateHandler();
    perf_profiler_ptr->PROFILER_END_CPU(PROFILER_PLANNING);

    //perf_profiler_ptr->PrintProfilerTimesPlanning();
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
            int left_most_x = left_lane_x_limits[2 * i];
            int right_most_x = left_lane_x_limits[2 * i + 1];

            if (left_most_x < IMG_WIDTH / 2 && right_most_x < IMG_WIDTH / 2)
            {
                left_lane_x[i] = (right_most_x + (right_most_x - LANE_WIDTH)) / 2;
            }
            /* Here the coordinate could be higher than the width of the image, we need to clamp */
            else if (left_most_x >= IMG_WIDTH / 2 && right_most_x >= IMG_WIDTH / 2)
            {
                left_lane_x[i] = (left_most_x + (left_most_x + LANE_WIDTH)) / 2;
            }

            if (left_lane_x[i] >= IMG_WIDTH)
            {
                left_lane_x[i] = IMG_WIDTH - 1;
            }
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
            int right_most_x = right_lane_x_limits[2 * i + 1];

            if (left_most_x < IMG_WIDTH / 2 && right_most_x < IMG_WIDTH / 2)
            {
                right_lane_x[i] = (right_most_x + (right_most_x - LANE_WIDTH)) / 2;
            }
            /* Here the coordinate could be higher than the width of the image, we need to clamp */
            else if (left_most_x >= IMG_WIDTH / 2 && right_most_x >= IMG_WIDTH / 2)
            {
                right_lane_x[i] = (left_most_x + (left_most_x + LANE_WIDTH)) / 2;
            }
            if (right_lane_x[i] >= IMG_WIDTH)
            {
                right_lane_x[i] = IMG_WIDTH - 1;
            }
        }
    }
}

int Planning::GetLaneCenter(int lane_idx)
{
    if (lane_idx == RIGHT_LANE)
    {
        for (int i = IMG_HEIGHT / CONTOUR_RES - 1; i > 0; i--)
        {
            if (right_lane_x[i] > 0)
            {
                /*if (obstacle)
                {
                    LogInfo("Depasire\n");
                    return right_lane_x[i] - 100;
                }*/
                return right_lane_x[i];
            }
        }
    }
    else if (lane_idx == LEFT_LANE)
    {
        for (int i = IMG_HEIGHT / CONTOUR_RES - 1; i > 0; i--)
        {
            if (left_lane_x[i] > 0)
            {
                return left_lane_x[i];
            }
        }
    }
    else if(lane_idx == BETWEEN_LANES)
    {
        int good_left_lane_x = -1;
        int good_right_lane_x = -1;
        for (int i = IMG_HEIGHT / CONTOUR_RES - 1; i > 0; i--)
        {
            if (left_lane_x[i] > 0)
            {
                good_left_lane_x = left_lane_x[i];
                break;
            }
        }
        for (int i = IMG_HEIGHT / CONTOUR_RES - 1; i > 0; i--)
        {
            if (right_lane_x[i] > 0)
            {
                good_right_lane_x = right_lane_x[i];
                break;
            }
        }
        if(good_left_lane_x>0&&good_right_lane_x>0)
        {
            return (good_left_lane_x+good_right_lane_x)/2;
        }
        else if(good_left_lane_x>0)
        {
            return good_left_lane_x;
        }
        else if(good_right_lane_x>0)
        {
            return good_right_lane_x;
        }
    }
    return 512;
}

bool Planning::IsObstacleOnLane()
{
    static int consecutive_detections = 0;
    static int consecutive_misses = 0;
    static bool change_lane = false;
    if (obstacle_limits[0] < obstacle_limits[1])
    {
        consecutive_detections++;
        consecutive_misses = 0;
    }
    else
    {
        consecutive_misses++;
        consecutive_detections = 0;
    }
    if (consecutive_detections > 10 && obstacle_limits[3] > OBSTACLE_DISTANCE_THRESH)
    {
        change_lane = true;
    }
    if (consecutive_misses > 20)
    {
        change_lane = false;
    }
    return change_lane;
}
