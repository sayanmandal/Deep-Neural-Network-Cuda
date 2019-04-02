#pragma once
#include "nn_layer.hh"


class LinearTanhLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;
	//Matrix T;
	//Matrix dZ;

	Matrix AT;
	Matrix WT;

	void initializeBiasWithZeros_tanh();
	void initializeWeightsRandomly_tanh();

	void computeAndStoreBackpropError_tanh(Matrix& dZ);
	void computeAndStoreLayerOutput_tanh(Matrix& A);
	void updateWeights_tanh(Matrix& dZ, float learning_rate);
	void updateBias_tanh(Matrix& dZ, float learning_rate);
  void TanhBackProp(Matrix& dZ);

public:
	LinearTanhLayer(std::string name, Shape W_shape);
	~LinearTanhLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;


};
