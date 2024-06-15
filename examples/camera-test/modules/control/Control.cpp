#include "Control.hpp"

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
