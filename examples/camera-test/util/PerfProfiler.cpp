#include "PerfProfiler.hpp"
#include <jetson-utils/cudaMappedMemory.h>

/*****************Performance profiling functions*****************/
/**
 * Begin a profiling query, before network is run
 */
void PerfProfiler::PROFILER_BEGIN(profilerQuery query, cudaStream_t stream)
{
    const uint32_t evt = query * 2;
    const uint32_t flag = (1 << query);

    CUDA(cudaEventRecord(mEventsGPU[evt], stream));
    timestamp(&mEventsCPU[evt]);

    mProfilerQueriesUsed |= flag;
    mProfilerQueriesDone &= ~flag;
}

/**
 * End a profiling query, after the network is run
 */
void PerfProfiler::PROFILER_END(profilerQuery query)
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
bool PerfProfiler::PROFILER_QUERY(profilerQuery query)
{
    const uint32_t flag = (1 << query);

    if (query == PROFILER_TOTAL_SEG)
    {
        // mProfilerTimes[PROFILER_TOTAL_SEG].x = 0.0f;
        // mProfilerTimes[PROFILER_TOTAL_SEG].y = 0.0f;

        for (uint32_t n = 0; n < PROFILER_TOTAL_SEG; n++)
        {
            if (PROFILER_QUERY((profilerQuery)n))
            {
                LogInfo("intra\n");
                mProfilerTimes[PROFILER_TOTAL_SEG].x += mProfilerTimes[n].x;
                mProfilerTimes[PROFILER_TOTAL_SEG].y += mProfilerTimes[n].y;
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

void PerfProfiler::PROFILER_BEGIN_CPU(profilerQuery query)
{
    const uint32_t evt = query * 2;

    timestamp(&mEventsCPU[evt]);
}

void PerfProfiler::PROFILER_END_CPU(profilerQuery query)
{
    const uint32_t evt = query * 2 + 1;

    timestamp(&mEventsCPU[evt]);
    timespec cpuTime;
    timeDiff(mEventsCPU[evt - 1], mEventsCPU[evt], &cpuTime);
    mProfilerTimes[query].x = timeFloat(cpuTime);
}

const char *profilerQueryToStr(profilerQuery query)
{
    switch (query)
    {
    case PROFILER_PREPROCESS_SEG:
        return "Pre-Process";
    case PROFILER_NETWORK_SEG:
        return "Network";
    case PROFILER_POSTPROCESS_SEG:
        return "Post-Process";
    case PROFILER_VISUALIZE_SEG:
        return "Visualize";
    case PROFILER_TOTAL_SEG:
        return "Total";
    }
    return nullptr;
}

/**
 * Print the profiler times (in millseconds).
 */
void PerfProfiler::PrintProfilerTimesPerception()
{
    LogInfo("\n");
    LogInfo("Timing Report Perception\n");
    LogInfo("------------------------------------------------------------------------------------------------\n");
    LogInfo("Segmentation network                              Detection network\n");
    LogInfo("------------------------------------------------------------------------------------------------\n");

    mProfilerTimes[PROFILER_TOTAL_SEG].x = 0.0f;
    mProfilerTimes[PROFILER_TOTAL_SEG].y = 0.0f;
    mProfilerTimes[PROFILER_TOTAL_DET].x = 0.0f;
    mProfilerTimes[PROFILER_TOTAL_DET].y = 0.0f;

    for (uint32_t n = 0; n <= PROFILER_TOTAL_DET; n++)
    {
        const profilerQuery query = (profilerQuery)n;

        if (PROFILER_QUERY(query))
        {
            if (n < PROFILER_TOTAL_SEG)
                LogInfo("%-12s  CPU %9.5fms  CUDA %9.5fms  CPU %9.5fms  CUDA %9.5fms\n",
                        profilerQueryToStr(query), mProfilerTimes[n].x, mProfilerTimes[n].y, mProfilerTimes[n + PROFILER_TOTAL_SEG + 1].x, mProfilerTimes[n + PROFILER_TOTAL_SEG + 1].y);
        }
    }
    for (uint32_t n = 0; n < PROFILER_TOTAL_SEG; n++)
    {
        mProfilerTimes[PROFILER_TOTAL_SEG].x += mProfilerTimes[n].x;
        mProfilerTimes[PROFILER_TOTAL_SEG].y += mProfilerTimes[n].y;
        mProfilerTimes[PROFILER_TOTAL_DET].x += mProfilerTimes[n + PROFILER_TOTAL_SEG].x;
        mProfilerTimes[PROFILER_TOTAL_DET].y += mProfilerTimes[n + PROFILER_TOTAL_SEG].y;
    }

    LogInfo("%-12s  CPU %9.5fms  CUDA %9.5fms  CPU %9.5fms  CUDA %9.5fms\n",
            profilerQueryToStr(PROFILER_TOTAL_SEG), mProfilerTimes[PROFILER_TOTAL_SEG].x, mProfilerTimes[PROFILER_TOTAL_SEG].y, mProfilerTimes[PROFILER_TOTAL_DET].x, mProfilerTimes[PROFILER_TOTAL_DET].y);
    LogInfo("------------------------------------------------------------------------------------------------\n\n");
}

void PerfProfiler::PrintProfilerTimesPlanning()
{
    LogInfo("\n");
    LogInfo("Timing Report Planning\n");
    LogInfo("------------------------------------------------------------------------------------------------\n");
    LogInfo("%-12s  CPU %9.5fms\n",
            profilerQueryToStr(PROFILER_TOTAL_SEG), mProfilerTimes[PROFILER_PLANNING].x);
    LogInfo("------------------------------------------------------------------------------------------------\n\n");
}

void PerfProfiler::PrintProfilerTimesControl()
{
    LogInfo("\n");
    LogInfo("Timing Report Control\n");
    LogInfo("------------------------------------------------------------------------------------------------\n");
    LogInfo("%-12s  CPU %9.5fms\n",
            profilerQueryToStr(PROFILER_TOTAL_SEG), mProfilerTimes[PROFILER_CONTROL].x);
    LogInfo("------------------------------------------------------------------------------------------------\n\n");
}
void PerfProfiler::PrintProfilerTimesMain()
{
    LogInfo("\n");
    LogInfo("Timing Report Main\n");
    LogInfo("------------------------------------------------------------------------------------------------\n");
    LogInfo("%-12s  CPU %9.5fms\n",
            profilerQueryToStr(PROFILER_TOTAL_SEG), mProfilerTimes[PROFILER_TOTAL].x);
    LogInfo("------------------------------------------------------------------------------------------------\n\n");
}

float PerfProfiler::GetLatency()
{
    return (float)mProfilerTimes[PROFILER_TOTAL].x;
}
float PerfProfiler::GetFPS()
{
    return 1000.0f / mProfilerTimes[PROFILER_TOTAL].x;
}

PerfProfiler::PerfProfiler()
{
    mProfilerQueriesUsed = 0;
    mProfilerQueriesDone = 0;

    memset(mEventsCPU, 0, sizeof(mEventsCPU));
    memset(mEventsGPU, 0, sizeof(mEventsGPU));
    memset(mProfilerTimes, 0, sizeof(mProfilerTimes));

    /*
     * create events for timing
     */
    for (int n = 0; n < PROFILER_TOTAL_DET * 2; n++)
        CUDA(cudaEventCreate(&mEventsGPU[n]));
}

PerfProfiler *PerfProfiler::getInstance()
{
    if (instancePtr == NULL)
    {
        instancePtr = new PerfProfiler();
        return instancePtr;
    }
    else
    {
        return instancePtr;
    }
}
