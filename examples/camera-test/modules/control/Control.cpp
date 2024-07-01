#include "Control.hpp"

void Control::Process(Planning *planning_module)
{
    perf_profiler_ptr->PROFILER_BEGIN_CPU(PROFILER_CONTROL);
    GetPlanningData(planning_module);
    SendSetpoints();
    perf_profiler_ptr->PROFILER_END_CPU(PROFILER_CONTROL);

    //perf_profiler_ptr->PrintProfilerTimesControl();
}

void Control::GetPlanningData(Planning *planning_module)
{
    speed_sp = planning_module->GetLongitudinalSetpoint();
    lateral_sp = planning_module->GetLateralSetpoint();
}

void Control::SendSetpoints()
{
    comm.lateralSetpoint = lateral_sp;
    comm.speedSetpoint = speed_sp;
    comm.publishMessage();
}
