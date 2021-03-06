#pragma once

#include "nn_layer.hh"

class tanhActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

	Matrix E;
	Matrix B;
	Matrix C;
	Matrix D;
	Matrix dE;
	Matrix dF;

public:
	tanhActivation(std::string name);
	~tanhActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
