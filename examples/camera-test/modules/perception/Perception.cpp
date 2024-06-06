#include "Perception.hpp"

/* Consecutive times a sign has to be detected to be trusted that it was actually detected */
#define DETECTION_FILTER_THRESH 4

/*Perception::Perception(){
	//FastScnn *segNetwork = new FastScnn(segModelPath);
	LogInfo("After serialization\n");
}*/

Perception::~Perception()
{
	// delete segNetwork;
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
	seg_network.Process();// 60ms 62ms(Paddle)
	seg_network.PostProcess(&classmap_ptr);

	//int lane_center = seg_network.getLaneCenter(1); // 57ms

	det_network.PreProcess(imgInput);
	det_network.Process();				  // run inference (22 ms)
	det_network.PostProcess(&detections); // nms (very fast)

	FilterDetections(detections);

#if VISUALIZATION_ENABLED
	OverlaySegImage(imgOutput,0);// 54 ms
	GetDetImage(imgInput);
	det_network.OverlayBBoxesOnVisImage(det_vis_image, DETECTION_ROI_W, DETECTION_ROI_H);
	OverlayDetImage(imgOutput);
	cudaDeviceSynchronize();
#endif
}

int Perception::GetDetection(Yolo::Detection *det)
{
	det->class_id = -1;
	if (detected_sign.frame_cnt >= DETECTION_FILTER_THRESH)
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

void Perception::GetDetImage(pixelType *img_input)
{
	getROIOfImage(img_input, det_vis_image, CAMERA_INPUT_W, CAMERA_INPUT_H, DETECTION_ROI_W, DETECTION_ROI_H);
}

void Perception::OverlayDetImage(pixelType *img_output)
{
	for (int y = 0; y < DETECTION_ROI_H; ++y)
	{
		for (int x = 0; x < DETECTION_ROI_W; ++x)
		{
			// Calculate position in source and destination arrays
			int srcPos = y * DETECTION_ROI_W + x;
			int dstPos = (y + 512) * 1024 + x;

			// Copy pixel
			img_output[dstPos] = det_vis_image[srcPos];
		}
	}
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