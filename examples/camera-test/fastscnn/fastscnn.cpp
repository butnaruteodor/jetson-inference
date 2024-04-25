#include "fastscnn.h"

static inline nvinfer1::Dims validateDims(const nvinfer1::Dims &dims)
{
	if (dims.nbDims == nvinfer1::Dims::MAX_DIMS)
		return dims;

	nvinfer1::Dims dims_out = dims;

	// TRT doesn't set the higher dims, so make sure they are 1
	for (int n = dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
		dims_out.d[n] = 1;

	return dims_out;
}

static inline void copyDims(Dims3 *dest, const nvinfer1::Dims *src)
{
	for (int n = 0; n < src->nbDims; n++)
		dest->d[n] = src->d[n];

	dest->nbDims = src->nbDims;
}

static inline nvinfer1::Dims shiftDims( const nvinfer1::Dims& dims )
{
	// TensorRT 7.0 requires EXPLICIT_BATCH flag for ONNX models,
	// which adds a batch dimension (4D NCHW), whereas historically
	// 3D CHW was expected.  Remove the batch dim (it is typically 1)
	nvinfer1::Dims out = dims;
	
	/*out.d[0] = dims.d[1];
	out.d[1] = dims.d[2];
	out.d[2] = dims.d[3];
	out.d[3] = 1;*/
	
	if( dims.nbDims == 1 )
		return out;
	
	for( int n=0; n < dims.nbDims; n++ )
		out.d[n] = dims.d[n+1];
	
	for( int n=dims.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
		out.d[n] = 1;
	
	out.nbDims -= 1;
	return out;
}

FastScnn::FastScnn(const std::string &engineFilename)
{
	mEngine = NULL;
	mInfer = NULL;
	mContext = NULL;
	mStream = NULL;
	mClassMap = NULL;
	// De-serialize engine from file
	std::ifstream engineFile(engineFilename, std::ios::binary);
	if (engineFile.fail())
	{
		printf("Failed to deserialize engine");
		return;
	}

	engineFile.seekg(0, std::ifstream::end);
	auto fsize = engineFile.tellg();
	engineFile.seekg(0, std::ifstream::beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	mInfer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	// mInfer = CREATE_INFER_RUNTIME(gLogger);
	mEngine = mInfer->deserializeCudaEngine(engineData.data(), fsize, nullptr);

	uGrid = nullptr;
	vGrid = nullptr;
}
int FastScnn::initEngine()
{
	// Context
	if (!mEngine)
		return false;

	mContext = mEngine->createExecutionContext();
	if (!mContext)
	{
		sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create execution context");
		return 0;
	}
	/*auto input_idx = mEngine->getBindingIndex("x");

	if (input_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
	auto input_dims = mContext->getBindingDimensions(input_idx);
	mContext->setBindingDimensions(input_idx, input_dims);
	auto input_size = sizeDims(input_dims, 1) * sizeof(float);

	auto output_idx = mEngine->getBindingIndex("save_infer_model/scale_0.tmp_0");
	if (output_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(output_idx) == nvinfer1::DataType::kFLOAT);
	auto output_dims = mContext->getBindingDimensions(output_idx);
	const size_t output_size = sizeDims(output_dims, 1) * sizeof(float);

	//second output
	auto output_idx1 = mEngine->getBindingIndex("save_infer_model/scale_1.tmp_0");
	if (output_idx1 == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(output_idx1) == nvinfer1::DataType::kFLOAT);
	auto output_dims1 = mContext->getBindingDimensions(output_idx1);
	const size_t output_size1 = sizeDims(output_dims1, 1) * sizeof(float);

	if (!cudaAllocMapped((void **)&outputBindCPU1, (void **)&outputBindGPU1, output_size1))
	{
		LogError("Could not allocate output1 CUDA\n");
		return false;
	}
	//end second output
	void *outputCUDA = NULL;
	void *outputCPU = NULL;
	if (!cudaAllocMapped((void **)&outputCPU, (void **)&outputCUDA, output_size))
	{
		LogError("Could not allocate output CUDA\n");
		return false;
	}

	void *inputCUDA = NULL;
	void *inputCPU = NULL;
	if (!cudaAllocMapped((void **)&inputCPU, (void **)&inputCUDA, input_size))
	{
		LogError("Could not allocate input CUDA\n");
		return false;
	}
	// CUDA(cudaMemset(inputCUDA, 0, input_size));

	// CUDA(cudaMemset(outputCUDA, 0, output_size));
	if (!cudaAllocMapped((void **)&mClassMap, 1024 * 512 * sizeof(uint8_t)))
		return false;

	if (cudaStreamCreate(&mStream) != cudaSuccess)
	{
		gLogError << "ERROR: cuda stream creation failed." << std::endl;
		return false;
	}

	const int bindingSize = 2 * sizeof(void *);
	mBindings = (void **)malloc(bindingSize);
	mBindings[0] = (float *)inputCUDA;
	mBindings[1] = (float *)outputCUDA;
	outputBind = (float *)outputCPU;
	inputBind = (float *)inputCPU;
	*/

	const int numBindings = mEngine->getNbBindings();

	std::vector<std::string> output_blobs;
	std::vector<std::string> input_blobs;
	output_blobs.push_back("save_infer_model/scale_0.tmp_0");
	output_blobs.push_back("save_infer_model/scale_1.tmp_0");
	input_blobs.push_back("x");
	//output_blobs.push_back("output");
	//input_blobs.push_back("input");

	const int numInputs = input_blobs.size();
	int mMaxBatchSize = 1;
	for (int n = 0; n < numInputs; n++)
	{
		const int inputIndex = mEngine->getBindingIndex(input_blobs[n].c_str());

		if (inputIndex < 0)
		{
			LogError("failed to find requested input layer %s in network\n", input_blobs[n].c_str());
			return false;
		}

		LogVerbose("binding to input %i %s  binding index:  %i\n", n, input_blobs[n].c_str(), inputIndex);
		nvinfer1::Dims inputDims = validateDims(mEngine->getBindingDimensions(inputIndex));
		inputDims = shiftDims(inputDims);
		const size_t inputSize = mMaxBatchSize * sizeDims(inputDims, 1) * sizeof(float);
		LogVerbose("binding to input %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, input_blobs[n].c_str(), mMaxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);

		// allocate memory to hold the input buffer
		void *inputCPU = NULL;
		void *inputCUDA = NULL;

		if (!cudaAllocMapped((void **)&inputCPU, (void **)&inputCUDA, inputSize))
		{
			LogError("failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
			return false;
		}

		layerInfo l;

		l.CPU = (float *)inputCPU;
		l.CUDA = (float *)inputCUDA;
		l.size = inputSize;
		l.name = input_blobs[n];
		l.binding = inputIndex;

		copyDims(&l.dims, &inputDims);
		mInputs.push_back(l);
	}

	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();

	for (int n = 0; n < numOutputs; n++)
	{
		const int outputIndex = mEngine->getBindingIndex(output_blobs[n].c_str());

		if (outputIndex < 0)
		{
			LogError("failed to find requested output layer %s in network\n", output_blobs[n].c_str());
			return false;
		}

		LogVerbose("binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

		nvinfer1::Dims outputDims = validateDims(mEngine->getBindingDimensions(outputIndex));
		outputDims = shiftDims(outputDims);  // change NCHW to CHW if EXPLICIT_BATCH set
		const size_t outputSize = mMaxBatchSize * sizeDims(outputDims, 1) * sizeof(float);
		LogVerbose("binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), mMaxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);

		// allocate output memory
		void *outputCPU = NULL;
		void *outputCUDA = NULL;

		// if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
		if (!cudaAllocMapped((void **)&outputCPU, (void **)&outputCUDA, outputSize))
		{
			LogError("failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
			return false;
		}

		layerInfo l;

		l.CPU = (float *)outputCPU;
		l.CUDA = (float *)outputCUDA;
		l.size = outputSize;
		l.name = output_blobs[n];
		l.binding = outputIndex;

		copyDims(&l.dims, &outputDims);
		mOutputs.push_back(l);
	}

	/*
	 * create list of binding buffers
	 */
	const int bindingSize = numBindings * sizeof(void *);

	mBindings = (void **)malloc(bindingSize);

	if (!mBindings)
	{
		LogError("failed to allocate %u bytes for bindings list\n", bindingSize);
		return false;
	}

	memset(mBindings, 0, bindingSize);

	for (uint32_t n = 0; n < GetInputLayers(); n++)
		mBindings[mInputs[n].binding] = mInputs[n].CUDA;

	for (uint32_t n = 0; n < GetOutputLayers(); n++)
		mBindings[mOutputs[n].binding] = mOutputs[n].CUDA;

	// find unassigned bindings and allocate them
	printf("numBindings: %d",numBindings);
	for (uint32_t n = 0; n < numBindings; n++)
	{
		if (mBindings[n] != NULL)
			continue;

		const size_t bindingSize = sizeDims(validateDims(mEngine->getBindingDimensions(n)),1) * mMaxBatchSize * sizeof(float);

		if (CUDA_FAILED(cudaMalloc(&mBindings[n], bindingSize)))
		{
			LogError("failed to allocate %zu bytes for unused binding %u\n", bindingSize, n);
			return false;
		}

		LogVerbose("allocated %zu bytes for unused binding %u\n", bindingSize, n);
	}

	if (!cudaAllocMapped((void **)&mClassMap, 1024 * 512 * sizeof(uint8_t)))
		return false;

	if (cudaStreamCreate(&mStream) != cudaSuccess)
	{
		gLogError << "ERROR: cuda stream creation failed." << std::endl;
		return false;
	}

	// Allocate the mapping arrays for undistortion/ipm
	cudaError_t u_malloc_err = cudaMalloc((void **)&uGrid, UV_GRID_COLS * sizeof(int));
	cudaError_t v_malloc_err = cudaMalloc((void **)&vGrid, UV_GRID_COLS * sizeof(int));

	if (u_malloc_err != cudaSuccess || v_malloc_err != cudaSuccess)
	{
		LogError("Could not allocate uGrid or vGrid\n");
		return false;
	}
	return true;
}
bool FastScnn::loadGrid()
{
	bool ret = true;
	int *uGridBuffer = nullptr;
	int *vGridBuffer = nullptr;

	uGridBuffer = (int *)malloc(UV_GRID_COLS * sizeof(int));
	vGridBuffer = (int *)malloc(UV_GRID_COLS * sizeof(int));

	std::ifstream infile_u("files/u_grid.bin", std::ios::binary);
	std::ifstream infile_v("files/v_grid.bin", std::ios::binary);

	if (!infile_u || !infile_v)
	{
		std::cout << "Cannot open file.\n";
		ret = false;
	}
	else
	{
		for (int j = 0; j < UV_GRID_COLS; ++j)
		{
			infile_u.read((char *)&uGridBuffer[j], sizeof(int));
		}
		for (int j = 0; j < UV_GRID_COLS; ++j)
		{
			infile_v.read((char *)&vGridBuffer[j], sizeof(int));
		}
		infile_u.close();
		infile_v.close();
	}

	cudaError_t cpy1_err = cudaMemcpy(this->uGrid, uGridBuffer, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);
	cudaError_t cpy2_err = cudaMemcpy(this->vGrid, vGridBuffer, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);

	if (cpy1_err != cudaSuccess || cpy2_err != cudaSuccess)
	{
		ret = false;
		LogError("Failed to copy to cuda mem\n");
	}

	free(uGridBuffer);
	free(vGridBuffer);

	return ret;
}
bool FastScnn::infer()
{
	if (!mContext->enqueueV2(mBindings, mStream, NULL))
	{
		LogError("failed to enqueue TensorRT context on device\n");
		return false;
	}
	return true;
}
bool FastScnn::process(uchar3 *image, uint32_t width, uint32_t height)
{
	warpImageK(image, (float *)mInputs[0].CUDA, uGrid, vGrid, width, height);
	// cudaDeviceSynchronize();
	if (!infer())
		return false;
	return true;
}
void FastScnn::toInputBinding(float3 *image_data)
{
	int width = 1024;
	int height = 512;
	int num_pixels = width * height;
	float *flat_array = new float[num_pixels * 3];

	int channel_size = num_pixels;

	// Normalize the image and arrange in NCHW format
	for (int i = 0; i < num_pixels; ++i)
	{
		// flat_array[i] = image_data[i].x;					   // R channel
		// flat_array[i + channel_size] = image_data[i].y;	   // G channel
		// flat_array[i + 2 * channel_size] = image_data[i].z; // B channel
	}

	// Copy data to the TensorRT engine's input binding
	cudaMemcpyAsync(mBindings[0], flat_array, num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice, mStream);

	// Cleanup
	delete[] flat_array;

	cudaStreamSynchronize(mStream);
}
void FastScnn::toInput(float *arr)
{
	cudaMemcpyAsync(mBindings[0], arr, 1024 * 512 * 3 * sizeof(float), cudaMemcpyHostToDevice, mStream);

	cudaStreamSynchronize(mStream);
}

bool FastScnn::classify()
{
	// retrieve scores
	float *scores = (float *)mBindings[1];

	const int s_w = OUT_IMG_W;
	const int s_h = OUT_IMG_H;
	const int s_c = 6;

	// find the argmax-classified class of each tile
	uint8_t *classMap = mClassMap;

	for (uint32_t y = 0; y < s_h; y++)
	{
		for (uint32_t x = 0; x < s_w; x++)
		{
			float p_max = -100000.0f;
			int c_max = 0;

			for (int c = 0; c < s_c; c++)
			{
				// check if this class score is higher
				const float p = scores[c * s_w * s_h + y * s_w + x];

				if (p > p_max)
				{
					p_max = p;
					c_max = c;
				}
			}

			classMap[y * s_w + x] = c_max;
		}
	}

	return true;
}
bool FastScnn::getRGB(pixelType *img, int middle_lane_x)
{
	for (uint32_t y = 0; y < OUT_IMG_H; y++)
	{
		for (uint32_t x = 0; x < OUT_IMG_W; x++)
		{
			int index = y * OUT_IMG_W + x;
			if ((int)mClassMap[index] == 0)
			{
				img[index].x = 128;
				img[index].y = 64;
				img[index].z = 128;
			}
			else if ((int)mClassMap[index] == 1)
			{
				img[index].x = 244;
				img[index].y = 35;
				img[index].z = 232;
			}
			else if ((int)mClassMap[index] == 2)
			{
				img[index].x = 70;
				img[index].y = 70;
				img[index].z = 70;
			}
			else if ((int)mClassMap[index] == 3)
			{
				img[index].x = 102;
				img[index].y = 102;
				img[index].z = 156;
			}
			else if ((int)mClassMap[index] == 4)
			{
				img[index].x = 190;
				img[index].y = 153;
				img[index].z = 153;
			}
			else if ((int)mClassMap[index] == 5)
			{
				img[index].x = 153;
				img[index].y = 153;
				img[index].z = 153;
			}
			if (x == middle_lane_x)
			{
				img[index].x = 0;
				img[index].y = 255;
				img[index].z = 0;
			}
		}
	}
}

void FastScnn::loopThroughClassmap(std::vector<int> &y_vals_lane, std::vector<int> &x_vals_lane, int classidx)
{
	obstacle.numPixels = 0;
	obstacle.smallest_x_obst = OUT_IMG_W;
	obstacle.biggest_x_obst = 0;
	obstacle.smallest_y_obst = OUT_IMG_H;
	obstacle.biggest_y_obst = 0;
	for (int y = 0; y < OUT_IMG_H; ++y)
	{ // Assuming height is 512
		for (int x = 0; x < OUT_IMG_W; ++x)
		{ // Assuming width is 1024
			int index = y * OUT_IMG_W + x;
			if (mClassMap[index] == classidx)
			{
				y_vals_lane.push_back(y);
				x_vals_lane.push_back(x);
			}
			else if (mClassMap[index] == OBSTACLE)
			{
				obstacle.numPixels++;
				if (x > obstacle.biggest_x_obst)
				{
					obstacle.biggest_x_obst = x;
				}
				else if (x < obstacle.smallest_x_obst)
				{
					obstacle.smallest_x_obst = x;
				}
				else if (y > obstacle.biggest_y_obst)
				{
					obstacle.biggest_y_obst = y;
				}
				else if (y < obstacle.smallest_y_obst)
				{
					obstacle.smallest_y_obst = y;
				}
			}
		}
	}
}

int FastScnn::getLaneCenter(int laneIdx)
{
	std::vector<int> y_vals_lane, x_vals_lane;
	int right_most_x = 0;
	int left_most_x = OUT_IMG_W; // Assuming width is 1024
	int middle_x, bottom_most_y;
	middle_x = -1;
	// Loop through the classMap to find right lane
	loopThroughClassmap(y_vals_lane, x_vals_lane, laneIdx);
	std::cout << y_vals_lane.size() << std::endl;
	if (!y_vals_lane.empty())
	{
		// Find the lowest y-coordinate for the lane
		bottom_most_y = *std::max_element(y_vals_lane.begin(), y_vals_lane.end());

		// Loop through a region around the bottom_most_y to find the left-most and right-most x
		for (int y = std::max(0, bottom_most_y - 200); y <= bottom_most_y; ++y)
		{
			for (int x = 0; x < OUT_IMG_W; ++x)
			{ // Assuming width is 1024
				int index = y * OUT_IMG_W + x;
				if (mClassMap[index] == laneIdx)
				{
					right_most_x = std::max(right_most_x, x);
					left_most_x = std::min(left_most_x, x);
				}
			}
		}

		middle_x = (right_most_x + left_most_x) / 2;
	}
	return middle_x;
}
int FastScnn::getParkingDirection(int offset)
{
	std::vector<int> y_vals_lane, x_vals_lane;
	int right_most_x = -1;
	loopThroughClassmap(y_vals_lane, x_vals_lane, CHARGING_PAD);
	if (!x_vals_lane.empty())
	{
		// Find the lowest y-coordinate for the lane
		right_most_x = *std::max_element(x_vals_lane.begin(), x_vals_lane.end()) + offset;
	}
	return right_most_x;
}
bool FastScnn::isObstacleOnLane(int dev)
{
	if (obstacle.numPixels > OBSTACLE_THRESH)
	{
		int center_x = (obstacle.smallest_x_obst + obstacle.biggest_x_obst) / 2;
		if (abs(center_x - OUT_IMG_W / 2) < dev)
		{
			return true;
		}
	}
	return false;
}
