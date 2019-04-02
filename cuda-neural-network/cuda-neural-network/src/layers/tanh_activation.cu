#include "tanh_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>


__global__ void expPlus(float* out, float* in, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = __expf(in[id]);
}


__global__ void expMinus(float* out, float* in, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = __expf(-in[id]);
}


__global__ void minusTanh(float* out, float* in1, float* in2, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = in1[id] - in2[id];
}


__global__ void plusTanh(float* out, float* in1, float* in2, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = in1[id] + in2[id];
}

__global__ void divideTanh(float* out, float* in1, float* in2, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = in1[id] / in2[id];
}

__global__ void multiplyTanh(float* out, float* in1, float* in2, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = in1[id] * in2[id];
}


__global__ void oneMinusTanh(float* out, float* in, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		out[id] = 1 - in[id];
}


__global__ void tanhActivationForward(float* Z, float* A,
										 int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = std::tanh(Z[index]);
	}
}

__global__ void tanhActivationBackprop(float* Z, float* dA, float* dZ,
										  int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
    float d = Z[index];
		dZ[index] = dA[index] * (1 - d * d);
	}
}

tanhActivation::tanhActivation(std::string name) {
	this->name = name;
}

tanhActivation::~tanhActivation()
{ }

Matrix& tanhActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	E.allocateMemoryIfNotAllocated(Z.shape);
	B.allocateMemoryIfNotAllocated(Z.shape);
	C.allocateMemoryIfNotAllocated(Z.shape);
	D.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	/*
	tanhActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y);
																*/

	expPlus<<<num_of_blocks, block_size>>>(E.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	expMinus<<<num_of_blocks, block_size>>>(B.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	minusTanh<<<num_of_blocks, block_size>>>(C.data_device.get(), E.data_device.get(), B.data_device.get(), C.shape.x * C.shape.y);
	plusTanh<<<num_of_blocks, block_size>>>(D.data_device.get(), E.data_device.get(), B.data_device.get(), D.shape.x * D.shape.y);
	divideTanh<<<num_of_blocks, block_size>>>(A.data_device.get(), C.data_device.get(), D.data_device.get(), A.shape.x * A.shape.y);

	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh forward propagation.");

	return A;
}

Matrix& tanhActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dE.allocateMemoryIfNotAllocated(Z.shape);
	dF.allocateMemoryIfNotAllocated(Z.shape);

	E.allocateMemoryIfNotAllocated(Z.shape);
	B.allocateMemoryIfNotAllocated(Z.shape);
	C.allocateMemoryIfNotAllocated(Z.shape);
	D.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
/*
	tanhActivationBackprop<<<num_of_blocks, block_size>>>(A.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
															 */

	expPlus<<<num_of_blocks, block_size>>>(E.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	expMinus<<<num_of_blocks, block_size>>>(B.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	minusTanh<<<num_of_blocks, block_size>>>(C.data_device.get(), E.data_device.get(), B.data_device.get(), C.shape.x * C.shape.y);
	plusTanh<<<num_of_blocks, block_size>>>(D.data_device.get(), E.data_device.get(), B.data_device.get(), D.shape.x * D.shape.y);
	divideTanh<<<num_of_blocks, block_size>>>(dE.data_device.get(), C.data_device.get(), D.data_device.get(), A.shape.x * A.shape.y);



	expPlus<<<num_of_blocks, block_size>>>(E.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	expMinus<<<num_of_blocks, block_size>>>(B.data_device.get(), Z.data_device.get(), Z.shape.x * Z.shape.y);
	minusTanh<<<num_of_blocks, block_size>>>(C.data_device.get(), E.data_device.get(), B.data_device.get(), C.shape.x * C.shape.y);
	plusTanh<<<num_of_blocks, block_size>>>(D.data_device.get(), E.data_device.get(), B.data_device.get(), D.shape.x * D.shape.y);
	divideTanh<<<num_of_blocks, block_size>>>(dF.data_device.get(), C.data_device.get(), D.data_device.get(), A.shape.x * A.shape.y);

	multiplyTanh<<<num_of_blocks, block_size>>>(dZ.data_device.get(), dE.data_device.get(), dF.data_device.get(), dZ.shape.x * dZ.shape.y);
	oneMinusTanh<<<num_of_blocks, block_size>>>(dZ.data_device.get(), dZ.data_device.get(), dZ.shape.x * dZ.shape.y);
	multiplyTanh<<<num_of_blocks, block_size>>>(dZ.data_device.get(), dA.data_device.get(), dZ.data_device.get(), dZ.shape.x * dZ.shape.y);
	
	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh back propagation");

	return dZ;
}
