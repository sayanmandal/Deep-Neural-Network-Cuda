#pragma once

#include "nn_layer.hh"

class softmaxActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	softmaxActivation(std::string name);
	~softmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
