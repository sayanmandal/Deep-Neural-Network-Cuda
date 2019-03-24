#include "softmax_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>


void printmatrix1(Matrix& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[i * m.shape.y + j] << " ";
		std::cout << std::endl;
	}

}


softmaxActivation::softmaxActivation(std::string name) {
  this->name = name;
}

softmaxActivation::~softmaxActivation()
{ }


  /*
  * blocks : cuSoftMaxP->rows
  * threads: cuSoftMaxP->cols
  * shared : sizeof(float) * cuSoftMaxP->cols * 2
  */
__global__ void g_getSoftMaxP(float* softMaxP, float* b, int cols){
  int bid = blockIdx.x;
	extern __shared__ float _share[];
	float * _max = _share;
	float * _sum = _share + blockDim.x;
	float* sp = softMaxP + bid * cols;
	_sum[threadIdx.x] = 0.0;
	_max[threadIdx.x] = -100000000.0;
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] += b[id];
			_max[threadIdx.x] = max(_max[threadIdx.x], sp[id]);
		}
	}
	__syncthreads();
	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			if(_max[threadIdx.x] < _max[threadIdx.x + skip])
			{
				_max[threadIdx.x] = _max[threadIdx.x + skip];
			}
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] -= _max[0];
			sp[id] = exp(sp[id]);
			_sum[threadIdx.x] += sp[id];
		}
	}
	__syncthreads();
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] /= _sum[0];
		}
	}
}



Matrix& softmaxActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);
  //int szy = Z.shape.y;
  dim3 block  = Z.shape.x;
  //dim3 thread = std::min(512, szy);
  //convert
  //Z.copyDeviceToHost();
  //printmatrix1(Z);

  int say = A.shape.y;
  int threads = std::min(512, say);
  g_getSoftMaxP<<<block, threads, sizeof(float) * threads * 2>>>(
  A.data_device.get(),
  Z.data_device.get(),
  A.shape.y);
  cudaStreamSynchronize(0);
  //A.copyDeviceToHost();
  //printmatrix1(A);
  /*
  std::cout << Z.shape.x << " " << Z.shape.y << std::endl;
  Z.copyDeviceToHost();
  for(int i = 0 ; i < Z.shape.x ; i++){
    for(int j = 0 ; j < Z.shape.y ; j++){
      std::cout << Z.data_host.get()[i * Z.shape.y + j] << " ";
    }
    std::cout << std::endl;
  }
  */

  //std::cout << A.shape.x << " " << A.shape.y << std::endl;
  //getLastCudaError("g_getSoftMaxP");

	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

	return A;
}


__global__ void softmaxActivationBackprop(float* Z, float* dA, float* dZ,
										  int Z_x_dim, int Z_y_dim){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < Z_x_dim * Z_y_dim){
    dZ[index] = dA[index];
  }

}


Matrix& softmaxActivation::backprop(Matrix& dA, float learning_rate) {
  dZ.allocateMemoryIfNotAllocated(Z.shape);
  /*
  dA.copyDeviceToHost();
  for(int i = 0 ; i < dA.shape.x ; i++){
    for(int j = 0 ; j < dA.shape.y ; j++){
      std::cout << dA.data_host.get()[i * dA.shape.y + j] << " ";
    }
    std::cout << std::endl;
  }
  */

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

	return dZ;

}
