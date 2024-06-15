#pragma once
#include "Planning.hpp"
#include "Comm.hpp"

class Control
{
    public:
    Control() : comm(SOCKET_COMM_TYPE){};
    /* Get the lateral setpoint and the longitudinal setpoint */
    void GetPlanningData(Planning* planning_module);
    /* Transmit the speed and lateral setpoints to the controllers */
    void SendSetpoints();

    private:
    int speed_sp;
    int lateral_sp;

    Comm comm;
};