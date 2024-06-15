#ifndef _FASTSCNN_
#define _FASTSCNN_

#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/timespec.h>

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

/**
 * Profiling queries
 * @see tensorNet::GetProfilerTime()
 * @ingroup tensorNet
 */
enum profilerQuery
{
	PROFILER_PREPROCESS = 0,
	PROFILER_NETWORK,
	PROFILER_POSTPROCESS,
	PROFILER_VISUALIZE,
	PROFILER_TOTAL,
};

class FastScnn
{
public:
	FastScnn(const std::string &engineFilename);
	~FastScnn();

	int InitEngine();
	void PreProcess(uchar3 *input_img);
	void Process();
	void PostProcess(uint8_t **classmap_ptr, int* class_pixels_indices);
	bool LoadGrid();


	/*****************Performance profiling functions*****************/
	/**
	 * Begin a profiling query, before network is run
	 */
	inline void PROFILER_BEGIN(profilerQuery query)
	{
		const uint32_t evt = query * 2;
		const uint32_t flag = (1 << query);

		CUDA(cudaEventRecord(mEventsGPU[evt], mStream));
		timestamp(&mEventsCPU[evt]);

		mProfilerQueriesUsed |= flag;
		mProfilerQueriesDone &= ~flag;
	}

	/**
	 * End a profiling query, after the network is run
	 */
	inline void PROFILER_END(profilerQuery query)
	{
		const uint32_t evt = query * 2 + 1;

		CUDA(cudaEventRecord(mEventsGPU[evt]));
		timestamp(&mEventsCPU[evt]);
		timespec cpuTime;
		timeDiff(mEventsCPU[evt - 1], mEventsCPU[evt], &cpuTime);
		mProfilerTimes[query].x = timeFloat(cpuTime);
	}

	/**
	 * Query the CUDA part of a profiler query.
	 */
	inline bool PROFILER_QUERY(profilerQuery query)
	{
		const uint32_t flag = (1 << query);

		if (query == PROFILER_TOTAL)
		{
			mProfilerTimes[PROFILER_TOTAL].x = 0.0f;
			mProfilerTimes[PROFILER_TOTAL].y = 0.0f;

			for (uint32_t n = 0; n < PROFILER_TOTAL; n++)
			{
				if (PROFILER_QUERY((profilerQuery)n))
				{
					mProfilerTimes[PROFILER_TOTAL].x += mProfilerTimes[n].x;
					mProfilerTimes[PROFILER_TOTAL].y += mProfilerTimes[n].y;
				}
			}

			return true;
		}
		else if (mProfilerQueriesUsed & flag)
		{
			if (!(mProfilerQueriesDone & flag))
			{
				const uint32_t evt = query * 2;
				float cuda_time = 0.0f;
				CUDA(cudaEventElapsedTime(&cuda_time, mEventsGPU[evt], mEventsGPU[evt + 1]));
				cudaThreadSynchronize();
				mProfilerTimes[query].y = cuda_time;
				mProfilerQueriesDone |= flag;
				mProfilerQueriesUsed &= ~flag;
			}

			return true;
		}

		return false;
	}

	const char *profilerQueryToStr(profilerQuery query)
	{
		switch (query)
		{
		case PROFILER_PREPROCESS:
			return "Pre-Process";
		case PROFILER_NETWORK:
			return "Network";
		case PROFILER_POSTPROCESS:
			return "Post-Process";
		case PROFILER_VISUALIZE:
			return "Visualize";
		case PROFILER_TOTAL:
			return "Total";
		}
		return nullptr;
	}

	/**
	 * Print the profiler times (in millseconds).
	 */
	inline void PrintProfilerTimes()
	{
		LogInfo("\n");
		LogInfo("------------------------------------------------\n");
		LogInfo("Timing Report\n");
		LogInfo("------------------------------------------------\n");

		for (uint32_t n = 0; n <= PROFILER_TOTAL; n++)
		{
			const profilerQuery query = (profilerQuery)n;

			if (PROFILER_QUERY(query))
				LogInfo("%-12s  CPU %9.5fms  CUDA %9.5fms\n", profilerQueryToStr(query), mProfilerTimes[n].x, mProfilerTimes[n].y);
		}

		LogInfo("------------------------------------------------\n\n");
	}

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
	cudaEvent_t mEventsGPU[PROFILER_TOTAL * 2];
	timespec mEventsCPU[PROFILER_TOTAL * 2];

	float2 mProfilerTimes[PROFILER_TOTAL + 1];
	uint32_t mProfilerQueriesUsed;
	uint32_t mProfilerQueriesDone;
};

#endif