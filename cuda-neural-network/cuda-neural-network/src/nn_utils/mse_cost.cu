#include "mse_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void msecost(float* predictions, float* target,
									   int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = (predictions[index] - target[index]) * (predictions[index] - target[index]);
		atomicAdd(cost,  partial_cost / size);
	}
}

__global__ void dMSECost(float* predictions, float* target, float* dY, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = 2 * (predictions[index] - target[index]);
	}
}

float MSECost::cost(Matrix predictions, Matrix target) {
	assert(predictions.shape.x == target.shape.x);
	//std::cout << predictions.shape.x << " " << predictions.shape.y << std::endl;

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	msecost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														  target.data_device.get(),
														  predictions.shape.x, cost);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}

Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dMSECost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
