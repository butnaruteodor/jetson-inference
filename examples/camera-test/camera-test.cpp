/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <signal.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "jetson-utils/videoSource.h"
#include "jetson-utils/videoOutput.h"

#include "Perception.hpp"
#include "Planning.hpp"
#include "Control.hpp"
#include "PerfProfiler.hpp"

using namespace std;
int fps,ms;
bool signal_recieved = false;
PerfProfiler *PerfProfiler::instancePtr = NULL;

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

//
// segmentation buffers
//

pixelType *imgOutput = NULL; // reference to one of the above three

int status;
int main(int argc, char **argv)
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

#if VISUALIZATION_ENABLED
	/*
	 * alloc space for visualization image
	 */
	if (!cudaAllocMapped(&imgOutput, make_int2(VIS_WINDOW_W, VIS_WINDOW_H)))
	{
		LogError("Ideas: Failed to allocate CUDA memory for out image\n");
		return 1;
	}
#endif
	/*
	 * attach signal handler
	 */
	if (signal(SIGINT, sig_handler) == SIG_ERR)
		LogError("can't catch SIGINT\n");

	/*
	 * create input stream
	 */
	videoSource *input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if (!input)
	{
		LogError("Ideas: Failed to create input stream\n");
		return 1;
	}

#if VISUALIZATION_ENABLED
	/*
	 * create output stream
	 */
	videoOutput *output = videoOutput::Create(cmdLine, ARG_POSITION(1));

	if (!output)
	{
		LogError("Ideas: Failed to create output stream\n");
		return 1;
	}
#endif

	Perception PerceptionModule;
	Planning PlanningModule;
	Control ControlModule;
	PerfProfiler *PerfProfiler = PerfProfiler::getInstance();

	while (!signal_recieved)
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		pixelType *imgInput = NULL;
		int status = 0;
		if (!input->Capture(&imgInput, &status))
		{
			if (status == videoSource::TIMEOUT)
				continue;

			break; // EOS
		}
		
		PerceptionModule.Process(imgInput, imgOutput);
		PlanningModule.Process(&PerceptionModule);
		ControlModule.Process(&PlanningModule);

#if VISUALIZATION_ENABLED
		if (output != NULL)
		{
			PlanningModule.OverlayLanePoints(imgOutput);
			output->Render(imgOutput, VIS_WINDOW_W, VIS_WINDOW_H);
			//  update status bar
			char str[256];
			sprintf(str, "Latency(ms): %d, FPS: %d", ms, fps);
			output->SetStatus(str);

			// check if the user quit
			if (!output->IsStreaming())
				break;
		}
#endif
		auto end_time = std::chrono::high_resolution_clock::now();
		ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
		fps = 1000.0f/ms;
	}
	/*
	 * destroy resources
	 */
	LogVerbose("Ideas: Shutting down...\n");

	SAFE_DELETE(input);
#if VISUALIZATION_ENABLED
	SAFE_DELETE(output);
	CUDA_FREE_HOST(imgOutput);
#endif
	LogVerbose("Ideas: Shutdown complete.\n");
	return 0;
}