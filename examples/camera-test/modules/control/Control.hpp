#pragma once
#include "Planning.hpp"
#include "Comm.hpp"

class Control
{
    public:
    Control() : comm(SOCKET_COMM_TYPE)
    {
        perf_profiler_ptr = PerfProfiler::getInstance();
    };
    void Process(Planning* planning_module);
    private:
    int speed_sp;
    int lateral_sp;

    Comm comm;
    PerfProfiler* perf_profiler_ptr;

    /* Get the lateral setpoint and the longitudinal setpoint */
    void GetPlanningData(Planning* planning_module);
    /* Transmit the speed and lateral setpoints to the controllers */
    void SendSetpoints();
};