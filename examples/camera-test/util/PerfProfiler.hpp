#pragma once
#include <cuda_runtime_api.h>
#include <jetson-utils/timespec.h>

/**
 * Profiling queries
 * @see tensorNet::GetProfilerTime()
 * @ingroup tensorNet
 */
enum profilerQuery
{
    PROFILER_PREPROCESS_SEG = 0,
    PROFILER_NETWORK_SEG,
    PROFILER_POSTPROCESS_SEG,
    PROFILER_VISUALIZE_SEG,
    PROFILER_TOTAL_SEG,
    PROFILER_PREPROCESS_DET,
    PROFILER_NETWORK_DET,
    PROFILER_POSTPROCESS_DET,
    PROFILER_VISUALIZE_DET,
    PROFILER_TOTAL_DET,
    PROFILER_PLANNING,
    PROFILER_CONTROL,
    PROFILER_TOTAL,
};

class PerfProfiler
{
private:
    /* Default constructor */
    PerfProfiler();

    static PerfProfiler *instancePtr;

    cudaEvent_t mEventsGPU[PROFILER_TOTAL_DET * 2];
    timespec mEventsCPU[PROFILER_TOTAL * 2];

    float2 mProfilerTimes[PROFILER_TOTAL + 1];
    uint32_t mProfilerQueriesUsed;
    uint32_t mProfilerQueriesDone;
public:

    /* Delete copy constructor */
    PerfProfiler(const PerfProfiler &obj) = delete;
    /* Singleton pattern */
    static PerfProfiler *getInstance();

    void PROFILER_BEGIN(profilerQuery query, cudaStream_t stream);
    void PROFILER_END(profilerQuery query);
    bool PROFILER_QUERY(profilerQuery query);

    void PROFILER_BEGIN_CPU(profilerQuery query);
    void PROFILER_END_CPU(profilerQuery query);

    void PrintProfilerTimesPerception();
    void PrintProfilerTimesPlanning();
    void PrintProfilerTimesControl();
    void PrintProfilerTimesMain();

    float GetLatency();
    float GetFPS();
};