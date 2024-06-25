#include "Perception.hpp"

/* Consecutive times a sign has to be detected to be trusted that it was actually detected */
#define DETECTION_FILTER_THRESH 3

Perception::~Perception()
{
	CUDA_FREE_HOST(det_vis_image);
	CUDA_FREE_HOST(left_lane_x_limits);
	CUDA_FREE_HOST(right_lane_x_limits);
	CUDA_FREE_HOST(charging_pad_center);
}

int Perception::InitModule()
{
	/* Initialize segmentation network */
	int status_seg = seg_network.InitEngine();
	if (!status_seg)
	{
		LogError("Perception: Failed to init fast scnn model\n");
		return 1;
	}

	/* Initialize detection network */
	int status_det = det_network.InitEngine();
	if (!status_det)
	{
		LogError("Perception: Failed to init yolo model\n");
		return 1;
	}
	return 0;
}
int Perception::RunPerception(pixelType *imgInput, pixelType *imgOutput)
{
	std::vector<Yolo::Detection> detections;

	seg_network.PreProcess(imgInput);
	seg_network.Process(); // 60ms 62ms(Paddle)
	InitializeLaneLimitsArray(left_lane_x_limits);
	InitializeLaneLimitsArray(right_lane_x_limits);
	InitializePadArray();
	InitializeObstacleLimitsArray();
	seg_network.PostProcess(&classmap_ptr, left_lane_x_limits, right_lane_x_limits, charging_pad_center, obstacle_limits);

	det_network.PreProcess(imgInput);
	det_network.Process();				  // run inference (22 ms)
	det_network.PostProcess(&detections); // nms (very fast)

	FilterDetections(detections);

#if VISUALIZATION_ENABLED
	OverlaySegImage(imgOutput, IMG_WIDTH / 2); // 54 ms
	GetDetImage(imgInput);
	OverlayBBoxesOnVisImage(det_vis_image, DETECTION_ROI_W, DETECTION_ROI_H);
	OverlayDetImage(imgOutput);
	cudaDeviceSynchronize();
#endif
}

int Perception::GetDetection(Yolo::Detection *det)
{
	det->class_id = -1;
	if (detected_sign.frame_cnt >= DETECTION_FILTER_THRESH && detected_sign.det.class_confidence > 0.8)
	{
		*det = detected_sign.det;
		return 0;
	}
	return 1;
}

uint8_t *Perception::GetSegmapPtr()
{
	return classmap_ptr;
}

int *Perception::GetLeftLaneXLimits()
{
	return left_lane_x_limits;
}

int *Perception::GetRightLaneXLimits()
{
	return right_lane_x_limits;
}

int *Perception::GetChargingPadCenter()
{
	return charging_pad_center;
}

int *Perception::GetObstacleLimitsArray()
{
    return obstacle_limits;
}

void Perception::GetDetImage(pixelType *img_input)
{
	getROIOfImage(img_input, det_vis_image, CAMERA_INPUT_W, CAMERA_INPUT_H, DETECTION_ROI_W, DETECTION_ROI_H);
}

void Perception::OverlayDetImage(pixelType *img_output)
{
	OverlayDetImagek(img_output, det_vis_image);
}

void Perception::OverlayBBoxesOnVisImage(uchar3 *out_image, int img_width, int img_height)
{
	uchar3 color = {0, 255, 0};
	Yolo::Detection det;
	GetDetection(&det);
	switch ((int)det.class_id)
	{
	case (YoloV3::STOP_SIGN_LABEL_ID):
	{
		uchar3 col = {255, 0, 0};
		color = col;
		break;
	}
	case (YoloV3::PARK_SIGN_LABEL_ID):
	{
		uchar3 col = {0, 0, 255};
		color = col;
		break;
	}
	case (YoloV3::CROSS_SIGN_LABEL_ID):
	{
		uchar3 col = {255, 255, 255};
		color = col;
		break;
	}
	case (YoloV3::CHARGE_SIGN_LABEL_ID):
	{
		uchar3 col = {0, 255, 0};
		color = col;
		break;
	}
	case (YoloV3::RED_LIGHT_LABEL_ID):
	{
		uchar3 col = {255, 0, 0};
		color = col;
		break;
	}
	case (YoloV3::YELLOW_LIGHT_LABEL_ID):
	{
		uchar3 col = {255, 255, 0};
		color = col;
		break;
	}
	}
	int offset_x = 50;
	int offset_y = 75;
	int bbox_x = (det.bbox[0] * img_width) / INPUT_W - offset_x;
	if (bbox_x < 0)
		bbox_x = 0;
	int bbox_y = (det.bbox[1] * img_height) / INPUT_H - offset_y;
	if (bbox_y < 0)
		bbox_y = 0;
	int bbox_w = (det.bbox[2] * img_width) / INPUT_W;
	int bbox_h = (det.bbox[3] * img_height) / INPUT_H;

	drawBoundingBox(out_image, img_width, img_height, bbox_x, bbox_y, bbox_w, bbox_h, color);
}

void Perception::FilterDetections(std::vector<Yolo::Detection> detections)
{
	int missed_threshold = 3;
	/* If a sign is detected and there were no other prior detections add the detected sign*/
	if (detections.size() != 0 && detected_sign.frame_cnt == 0)
	{
		detected_sign.det = detections.at(0); // for now take the first detection
		detected_sign.frame_cnt++;
		detected_sign.miss_cnt = 0;
	}
	/* If a sign is detected */
	else if (detections.size() != 0)
	{
		/* If its the same sign as last frame */
		if (detections.at(0).class_id == detected_sign.det.class_id)
		{
			/* Increment count */
			detected_sign.frame_cnt++;
			detected_sign.miss_cnt = 0;
			detected_sign.det = detections.at(0);
		}
		/* If its not count as a missed detection */
		else
		{
			detected_sign.miss_cnt++;
		}
	}
	/* Missed a detection */
	else if (detections.size() == 0)
	{
		detected_sign.miss_cnt++;
		/* Assume its a false negative(the sign is there but it wasnt detected) */
		detected_sign.frame_cnt++;
	}
	/* If havent detected a sign for missed_threshold frames it must mean there is no sign*/
	if (detected_sign.miss_cnt >= missed_threshold)
	{
		detected_sign.miss_cnt = 0;
		detected_sign.frame_cnt = 0;
	}
}

void Perception::OverlaySegImage(pixelType *img, int middle_lane_x)
{
	OverlaySegImageK(img, middle_lane_x, classmap_ptr, OUT_IMG_W, OUT_IMG_H);
}

void Perception::InitializeLaneLimitsArray(int *classes_extremities_x)
{
	for (int i = 0; i < IMG_HEIGHT / CONTOUR_RES; i++)
	{
		classes_extremities_x[2 * i] = IMG_WIDTH;
		classes_extremities_x[2 * i + 1] = 0;
	}
}

void Perception::InitializePadArray()
{
	charging_pad_center[0] = 1024;
	charging_pad_center[1] = 0;
	charging_pad_center[2] = 512;
	charging_pad_center[3] = 0;
}

void Perception::InitializeObstacleLimitsArray()
{
	obstacle_limits[0] = 1024;
	obstacle_limits[1] = 0;
	obstacle_limits[2] = 512;
	obstacle_limits[3] = 0;
}
