#ifndef _FASTSCNN_
#define _FASTSCNN_

#include <jetson-utils/cudaMappedMemory.h>

#include "logger.h"
#include "ipm.h"
#include "argmax.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <vector>
#include <algorithm>

#define OUT_IMG_W 1024
#define OUT_IMG_H 512
#define IN_IMG_W 1920
#define IN_IMG_H 1080
#define UV_GRID_COLS 524288
#define SEG_MAP_W 1024
#define SEG_MAP_H 512

#define VOID 0
#define RIGHT_LANE 1
#define LEFT_LANE 2
#define MARKINGS 3
#define CHARGING_PAD 4
#define OBSTACLE 5

#define OBSTACLE_THRESH 10000

#if NV_TENSORRT_MAJOR >= 6
typedef nvinfer1::Dims3 Dims3;
#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]
#endif

typedef uchar3 pixelType; // this can be uchar3, uchar4, float3, float4

using sample::gLogError;
using sample::gLogInfo;

struct Obstacle
{
	int numPixels;
	int smallest_x_obst;
	int biggest_x_obst;
	int smallest_y_obst;
	int biggest_y_obst;
};

class FastScnn
{
public:
	FastScnn(const std::string &engineFilename);
	~FastScnn();

	int InitEngine();
	void PreProcess(uchar3 *input_img);
	void Process();
	void PostProcess(uint8_t **classmap_ptr, int* left_lane_x_limits, int* right_lane_x_limits, int* charging_pad_center, int* obstacle_limits);
	bool LoadGrid();

	cudaStream_t GetStream();

	/**
	 * Retrieve the number of input layers to the network.
	 */
	inline uint32_t GetInputLayers() const { return mInputs.size(); }

	/**
	 * Retrieve the number of output layers to the network.
	 */
	inline uint32_t GetOutputLayers() const { return mOutputs.size(); }

	uint8_t *mClassMap;
	int *uGrid;
	int *vGrid;
	void **mBindings;

	struct layerInfo
	{
		std::string name;
		Dims3 dims;
		uint32_t size;
		uint32_t binding;
		float *CPU;
		float *CUDA;
	};

	std::vector<layerInfo> mInputs;
	std::vector<layerInfo> mOutputs;
	
protected:
	nvinfer1::ICudaEngine *mEngine;
	nvinfer1::IRuntime *mInfer;
	nvinfer1::IExecutionContext *mContext;
	cudaStream_t mStream;

	Obstacle obstacle;
};

#endif