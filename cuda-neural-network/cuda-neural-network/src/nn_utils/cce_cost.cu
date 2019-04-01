#include "cce_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

void printmatrix2(Matrix& m){
	for(int i = 0 ; i < m.shape.x; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[j * m.shape.x + i] << " ";
		std::cout << std::endl;
	}

}


__global__
void dCategoricalCrossEntropyCost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = (predictions[index] - target[index]);
	}
}


__global__
void categoricalCrossEntropyCost(float* predictions, float* target,
									   int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index]);
		atomicAdd(cost, - partial_cost / size);
	}
}




float CCECost::cost(Matrix predictions, Matrix target) {
	assert(predictions.shape.x == target.shape.x && predictions.shape.y == target.shape.y);

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;
	/*
	for(int i = 0 ; i < target.shape.x ; i++){
		for(int j = 0 ; j < target.shape.y ; j++){
			std::cout << target.data_host.get()[i * target.shape.y + j] << " ";
		}
		std::cout << std::endl;
	}
	*/
	//predictions.copyDeviceToHost();
	//printmatrix2(predictions);


	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x * predictions.shape.y + block_size.x - 1) / block_size.x);
	categoricalCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														  target.data_device.get(),
														  predictions.shape.x * predictions.shape.y, cost);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute categorical cross entropy cost.");

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}



Matrix CCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x && predictions.shape.y == target.shape.y);
	//std::cout << predictions.
	//dA.copyDeviceToHost();



	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x * predictions.shape.y + block_size.x - 1) / block_size.x);
	dCategoricalCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x * predictions.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for categorical cross entropy.");
	//std::cout << dY.shape.x << " " << dY.shape.y << std::endl;
	return dY;
}
