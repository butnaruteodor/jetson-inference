#include "Fastscnn.hpp"
#include "ProjectPaths.h"

static inline size_t sizeDims(const nvinfer1::Dims &dims, const size_t elementSize = 1)
{
	size_t sz = dims.d[0];

	for (int n = 1; n < dims.nbDims; n++)
		sz *= dims.d[n];

	return sz * elementSize;
}

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

static inline nvinfer1::Dims shiftDims(const nvinfer1::Dims &dims)
{
	// TensorRT 7.0 requires EXPLICIT_BATCH flag for ONNX models,
	// which adds a batch dimension (4D NCHW), whereas historically
	// 3D CHW was expected.  Remove the batch dim (it is typically 1)
	nvinfer1::Dims out = dims;

	/*out.d[0] = dims.d[1];
	out.d[1] = dims.d[2];
	out.d[2] = dims.d[3];
	out.d[3] = 1;*/

	if (dims.nbDims == 1)
		return out;

	for (int n = 0; n < dims.nbDims; n++)
		out.d[n] = dims.d[n + 1];

	for (int n = dims.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
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
	std::cout << mEngine << std::endl;
	uGrid = nullptr;
	vGrid = nullptr;
}
FastScnn::~FastScnn()
{
	if (mEngine)
	{
		mEngine->destroy();
	}
	if (mInfer)
	{
		mInfer->destroy();
	}
	if (mContext)
	{
		mContext->destroy();
	}
	if (mClassMap)
	{
		CUDA_FREE_HOST(mClassMap);
	}

	for (size_t n = 0; n < mInputs.size(); n++)
	{
		CUDA_FREE(mInputs[n].CUDA);
	}

	for (size_t n = 0; n < mOutputs.size(); n++)
		CUDA_FREE_HOST(mOutputs[n].CPU);

	free(mBindings);

	cudaStreamDestroy(mStream);
}

int FastScnn::InitEngine()
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

	const int numBindings = mEngine->getNbBindings();

	std::vector<std::string> output_blobs;
	std::vector<std::string> input_blobs;
	output_blobs.push_back("save_infer_model/scale_0.tmp_0");
	output_blobs.push_back("save_infer_model/scale_1.tmp_0");
	input_blobs.push_back("x");

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
		outputDims = shiftDims(outputDims); // change NCHW to CHW if EXPLICIT_BATCH set
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
	printf("numBindings: %d", numBindings);
	for (uint32_t n = 0; n < numBindings; n++)
	{
		if (mBindings[n] != NULL)
			continue;

		const size_t bindingSize = sizeDims(validateDims(mEngine->getBindingDimensions(n)), 1) * mMaxBatchSize * sizeof(float);

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

	int status = LoadGrid();
	if (!status)
	{
		LogError("FastScnn: Failed to load grid\n");
		return 1;
	}
	return true;
}

void FastScnn::PreProcess(uchar3 *input_img)
{
	warpImageK(input_img, (float *)mInputs[0].CUDA, uGrid, vGrid, IN_IMG_W, IN_IMG_H); // 3ms
}

void FastScnn::Process()
{
	if (!mContext->enqueueV2(mBindings, mStream, NULL))
	{
		LogError("Fastscnn: Failed to enqueue TensorRT context on device\n");
	}
	//CUDA(cudaStreamSynchronize(mStream));
}

void FastScnn:: PostProcess(uint8_t **classmap_ptr, int* left_lane_x_limits, int* right_lane_x_limits, int* charging_pad_center, int* obstacle_limits)
{
	generateClassMap((float *)mOutputs[0].CUDA, mClassMap, left_lane_x_limits, right_lane_x_limits, charging_pad_center, obstacle_limits); // 1ms

	*classmap_ptr = mClassMap;
}

bool FastScnn::LoadGrid()
{
	bool ret = true;
	int *uGridBuffer = nullptr;
	int *vGridBuffer = nullptr;

	uGridBuffer = (int *)malloc(UV_GRID_COLS * sizeof(int));
	vGridBuffer = (int *)malloc(UV_GRID_COLS * sizeof(int));

	std::ifstream infile_u(ugrid_path, std::ios::binary);
	std::ifstream infile_v(vgrid_path, std::ios::binary);

	if (!infile_u || !infile_v)
	{
		LogError("Fastscnn: Failed to open calibration files\n");
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
		LogError("Fastscnn: Failed to copy uGrid/vGrid to cuda mem\n");
	}

	free(uGridBuffer);
	free(vGridBuffer);

	return ret;
}

cudaStream_t FastScnn::GetStream()
{
    return mStream;
}
